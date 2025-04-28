import glob
import os
import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import deque

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small

from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from .image_utils import postprocess_image, forward_backward_consistency_check
from .models.utils import get_nn_latent
from .image_filter import SimilarImageFilter


class VideoPipeline:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_tensor = None
        self.prev_x_t_latent = None
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.cached_x_t_latent = deque(maxlen=4)

        self.inference_time_ema = 0

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = 'lcm',
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)
        self.null_prompt_embeds = encoder_output[1]

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.randn_noise = self.init_noise[:1].clone()
        self.warp_noise = self.init_noise[:1].clone()

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        self.alpha_prod_t_sqrt = alpha_prod_t_sqrt
        self.beta_prod_t_sqrt = beta_prod_t_sqrt

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )

        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)
        self.null_prompt_embeds = encoder_output[1]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        alpha_prod_t_sqrt = self.alpha_prod_t_sqrt[t_index]
        beta_prod_t_sqrt = self.beta_prod_t_sqrt[t_index]

        noisy_samples = (
            alpha_prod_t_sqrt * original_samples + beta_prod_t_sqrt * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        if idx is None:
            idx = 0

        c_skip = self.c_skip[idx]
        c_out = self.c_out[idx]

        x_0_pred = c_skip * x_t_latent_batch + c_out * model_pred_batch

        return x_0_pred

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx is None:
            idx = 0

        if isinstance(t_list, list):
            t_list = torch.tensor(t_list, device=self.device)

        if self.guidance_scale > 1.0:
            if self.cfg_type == "self":
                x_t_latent_cond = x_t_latent
                x_t_latent_uncond = x_t_latent
            elif self.cfg_type == "initialize":
                x_t_latent_cond = x_t_latent
                x_t_latent_uncond = self.init_noise[: self.frame_bff_size]
            elif self.cfg_type == "full":
                x_t_latent_cond = x_t_latent
                x_t_latent_uncond = x_t_latent

            if self.cfg_type == "self":
                model_pred = self.unet(
                    x_t_latent_cond,
                    t_list,
                    encoder_hidden_states=self.prompt_embeds,
                ).sample

                model_pred_uncond = self.unet(
                    x_t_latent_uncond,
                    t_list,
                    encoder_hidden_states=self.null_prompt_embeds,
                ).sample

                model_pred = model_pred_uncond + self.guidance_scale * (
                    model_pred - model_pred_uncond
                )
            elif self.cfg_type == "initialize":
                model_pred = self.unet(
                    x_t_latent_cond,
                    t_list,
                    encoder_hidden_states=self.prompt_embeds,
                ).sample

                model_pred_uncond = self.unet(
                    x_t_latent_uncond,
                    t_list,
                    encoder_hidden_states=self.null_prompt_embeds,
                ).sample

                model_pred = model_pred_uncond + self.guidance_scale * (
                    model_pred - model_pred_uncond
                )
            elif self.cfg_type == "full":
                model_pred = self.unet(
                    x_t_latent,
                    t_list,
                    encoder_hidden_states=self.prompt_embeds,
                ).sample
        else:
            model_pred = self.unet(
                x_t_latent,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
            ).sample

        x_0_pred = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return x_0_pred, model_pred

    def norm_noise(self, noise):
        mean = torch.mean(noise, dim=(2, 3), keepdim=True)
        std = torch.std(noise, dim=(2, 3), keepdim=True)
        return (noise - mean) / (std + 1e-6)

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.dtype)
        latents = self.vae.encode(image_tensors).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor
        return latents

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        x_0_pred_out = x_0_pred_out / self.pipe.vae.config.scaling_factor
        image = self.vae.decode(x_0_pred_out).sample
        return image

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        if self.use_denoising_batch:
            if self.cfg_type == "initialize":
                x_t_latent_cond = x_t_latent
                x_t_latent_uncond = self.init_noise[: self.frame_bff_size]
                x_t_latent = torch.cat([x_t_latent_uncond, x_t_latent_cond], dim=0)
            elif self.cfg_type == "full":
                x_t_latent_cond = x_t_latent
                x_t_latent_uncond = x_t_latent
                x_t_latent = torch.cat([x_t_latent_uncond, x_t_latent_cond], dim=0)

            x_0_pred_batch, _ = self.unet_step(
                x_t_latent,
                self.sub_timesteps_tensor,
            )

            if self.cfg_type == "initialize":
                x_0_pred_batch = x_0_pred_batch[self.frame_bff_size :]
            elif self.cfg_type == "full":
                x_0_pred_batch = x_0_pred_batch[self.batch_size :]

            x_0_pred_batch = x_0_pred_batch.view(
                self.denoising_steps_num,
                self.frame_bff_size,
                4,
                self.latent_height,
                self.latent_width,
            )
        else:
            x_0_pred_batch = []
            for i in range(self.denoising_steps_num):
                x_0_pred, _ = self.unet_step(
                    x_t_latent,
                    self.sub_timesteps_tensor[i * self.frame_bff_size],
                    i,
                )
                x_0_pred_batch.append(x_0_pred)
            x_0_pred_batch = torch.stack(x_0_pred_batch)

        return x_0_pred_batch

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        if x is None:
            return self.txt2img()

        if isinstance(x, PIL.Image.Image):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        x = x.to(device=self.device, dtype=self.dtype)
        x = F.interpolate(x, size=(self.height, self.width), mode="bilinear")

        if self.similar_image_filter:
            if self.prev_image_tensor is not None:
                if self.similar_filter.is_similar(x, self.prev_image_tensor):
                    return self.prev_image_result

        x_t_latent = self.encode_image(x)

        if self.do_add_noise:
            x_t_latent = self.add_noise(x_t_latent, self.randn_noise, 0)

        x_0_pred_batch = self.predict_x0_batch(x_t_latent)

        x_0_pred_out = x_0_pred_batch[-1, 0]
        x_0_pred_out = self.decode_image(x_0_pred_out)

        if self.similar_image_filter:
            self.prev_image_tensor = x
            self.prev_image_result = x_0_pred_out

        return x_0_pred_out

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = self.init_noise[:batch_size]
        x_0_pred_batch = self.predict_x0_batch(x_t_latent)
        x_0_pred_out = x_0_pred_batch[-1, 0]
        x_0_pred_out = self.decode_image(x_0_pred_out)
        return x_0_pred_out

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = self.init_noise[:batch_size]
        x_0_pred_batch = self.predict_x0_batch(x_t_latent)
        x_0_pred_out = x_0_pred_batch[-1, 0]
        x_0_pred_out = self.decode_image(x_0_pred_out)
        return x_0_pred_out