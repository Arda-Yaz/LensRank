"""Model, data and experiment configuration."""

from dataclasses import dataclass
import torch


@dataclass
class SuiteConfig:
    data_dir: str = 'data'
    run_dir: str = 'artifacts/fresh_runs'
    model_name: str = 'ViT-B-16'
    pretrained: str = 'openai'
    epochs: int = 5
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    seed: int = 42
    checkpoint_every_batches: int = 100
    fast_dev_run: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


EXPERIMENTS = {
    "clip_zero_shot": {"strategy": "zero_shot", "lr": 0.0, "epochs": 0},
    "clip_projection_tuning": {"strategy": "projection", "lr": 1e-4},
    "openclip_full_finetune": {"strategy": "full", "lr": 1e-6},
    "openclip_gradual_unfreeze": {"strategy": "gradual", "lr": 5e-6},
    "openclip_lora": {
        "strategy": "lora",
        "lr": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
    },
}
