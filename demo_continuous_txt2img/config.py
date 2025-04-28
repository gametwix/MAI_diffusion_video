from dataclasses import dataclass, field
from typing import List, Literal

import torch
import os

SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "False") == "True"


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 7861
    workers: int = 1
    model_id_or_path: str = os.environ.get("MODEL", "KBlueLeaf/kohaku-v2.1")
    lora_dict: dict = None
    # LCM-LORA model
    lcm_lora_id: str = os.environ.get("LORA", "latent-consistency/lcm-lora-sdv1-5")
    # TinyVAE model
    vae_id: str = os.environ.get("VAE", "madebyollin/taesd")
    # Device to use
    device: torch.device = torch.device("cuda")
    # Data type
    dtype: torch.dtype = torch.float16
    # acceleration
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers"

    warmup: int = 10
    use_safety_checker: bool = SAFETY_CHECKER
