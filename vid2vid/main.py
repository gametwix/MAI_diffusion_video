import os
import sys
import time
from typing import Literal, Dict, Optional

import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.wrapper import VideoPipelineWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str,
    prompt: str,
    output_dir: str = os.path.join(CURRENT_DIR, "outputs"),
    model_id: str = "Jiali/stable-diffusion-1.5",
    scale: float = 1.0,
    guidance_scale: float = 1.0,
    diffusion_steps: int = 4,
    noise_strength: float = 0.4,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    use_cached_attn: bool = True,
    use_feature_injection: bool = True,
    feature_injection_strength: float = 0.8,
    feature_similarity_threshold: float = 0.98,
    cache_interval: int = 4,
    cache_maxframes: int = 1,
    use_tome_cache: bool = True,
    do_add_noise: bool = True,
    enable_similar_image_filter: bool = False,
    seed: int = 2,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    height = int(video.shape[1] * scale)
    width = int(video.shape[2] * scale)

    init_step = int(50 * (1 - noise_strength))
    interval = int(50 * noise_strength) // diffusion_steps
    t_index_list = [init_step + i * interval for i in range(diffusion_steps)]


    stream = VideoPipelineWrapper(
        model_id_or_path=model_id,
        mode="img2img",
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        use_cached_attn=use_cached_attn,
        use_feature_injection=use_feature_injection,
        feature_injection_strength=feature_injection_strength,
        feature_similarity_threshold=feature_similarity_threshold,
        cache_interval=cache_interval,
        cache_maxframes=cache_maxframes,
        use_tome_cache=use_tome_cache,
        seed=seed,
    )
    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
    )

    if any(word in prompt for word in ['pixelart', 'pixel art', 'Pixel art', 'PixArFK']):
        stream.stream.load_lora("./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name='pixelart')
        stream.stream.pipe.set_adapters(["lcm", "pixelart"], adapter_weights=[1.0, 1.0])
        print("Use LORA: pixelart in ./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors")
    elif any(word in prompt for word in ['lowpoly', 'low poly', 'Low poly']):
        stream.stream.load_lora("./lora_weights/low_poly.safetensors", adapter_name='lowpoly')
        stream.stream.pipe.set_adapters(["lcm", "lowpoly"], adapter_weights=[1.0, 1.0])
        print("Use LORA: lowpoly in ./lora_weights/low_poly.safetensors")
    elif any(word in prompt for word in ['Claymation', 'claymation']):
        stream.stream.load_lora("./lora_weights/Claymation.safetensors", adapter_name='claymation')
        stream.stream.pipe.set_adapters(["lcm", "claymation"], adapter_weights=[1.0, 1.0])
        print("Use LORA: claymation in ./lora_weights/Claymation.safetensors")
    elif any(word in prompt for word in ['crayons', 'Crayons', 'crayons doodle', 'Crayons doodle']):
        stream.stream.load_lora("./lora_weights/doodle.safetensors", adapter_name='crayons')
        stream.stream.pipe.set_adapters(["lcm", "crayons"], adapter_weights=[1.0, 1.0])
        print("Use LORA: crayons in ./lora_weights/doodle.safetensors")
    elif any(word in prompt for word in ['sketch', 'Sketch', 'pencil drawing', 'Pencil drawing']):
        stream.stream.load_lora("./lora_weights/Sketch_offcolor.safetensors", adapter_name='sketch')
        stream.stream.pipe.set_adapters(["lcm", "sketch"], adapter_weights=[1.0, 1.0])
        print("Use LORA: sketch in ./lora_weights/Sketch_offcolor.safetensors")
    elif any(word in prompt for word in ['oil painting', 'Oil painting']):
        stream.stream.load_lora("./lora_weights/bichu-v0612.safetensors", adapter_name='oilpainting')
        stream.stream.pipe.set_adapters(["lcm", "oilpainting"], adapter_weights=[1.0, 1.0])
        print("Use LORA: oilpainting in ./lora_weights/bichu-v0612.safetensors")

    video_result = torch.zeros(video.shape[0], height, width, 3)

    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    inference_time = []
    for i in tqdm(range(video.shape[0])):
        iteration_start_time = time.time()
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)
        iteration_end_time = time.time()
        inference_time.append(iteration_end_time -iteration_start_time )
    print(f'Avg time: {sum(inference_time[20:])/len(inference_time[20:])}')

    video_result = video_result * 255
    prompt_txt = prompt.replace(' ', '-')
    input_vid = input.split('/')[-1]
    output = os.path.join(output_dir, f"{input_vid.rsplit('.', 1)[0]}_{prompt_txt}.{input_vid.rsplit('.', 1)[1]}")
    write_video(output, video_result, fps=fps)


if __name__ == "__main__":
    fire.Fire(main)
