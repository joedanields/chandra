from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    MODEL_CHECKPOINT: str = "datalab-to/chandra"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 12384
    TORCH_ATTN: str | None = None
    # Allow overriding dtype via environment variable; defaults to bfloat16
    TORCH_DTYPE_NAME: str = "bfloat16"

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    @computed_field
    @property
    def TORCH_DTYPE(self) -> torch.dtype:
        name = (self.TORCH_DTYPE_NAME or "bfloat16").lower()
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        if name in {"fp32", "float32", "f32"}:
            return torch.float32
        # Fallback to bfloat16 if unrecognized
        return torch.bfloat16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
