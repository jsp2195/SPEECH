from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from rich.console import Console

from personalized_hearing_enhancement.data.download import download_datasets
from personalized_hearing_enhancement.evaluation.demo_audio import main as demo_audio_main
from personalized_hearing_enhancement.training.train import run_training
from personalized_hearing_enhancement.video.video_pipeline import create_comparison_video

app = typer.Typer(help="Audiogram-conditioned speech enhancement system")
console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@app.command()
def prepare_data(config: str = "personalized_hearing_enhancement/configs/default.yaml") -> None:
    cfg = OmegaConf.load(config)
    set_seed(int(cfg.seed))
    download_datasets(cfg.paths.data_cache)


@app.command()
def train(
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    model_type: str = typer.Option("baseline", help="baseline or conditioned"),
) -> None:
    cfg = OmegaConf.load(config)
    set_seed(int(cfg.seed))
    run_training(config, model_type)


@app.command()
def demo_audio(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str = "20,25,30,45,60,65,70,75",
) -> None:
    cfg = OmegaConf.load(config)
    set_seed(int(cfg.seed))
    demo_audio_main(input_wav, config, baseline_ckpt, conditioned_ckpt, audiogram, "outputs/demo_audio")


@app.command()
def process_video(
    input: str = typer.Option(..., "--input", help="Input MP4"),
    audiogram: str = typer.Option("20,25,30,45,60,65,70,75", help="8 comma-separated dB losses"),
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
) -> None:
    cfg = OmegaConf.load(config)
    set_seed(int(cfg.seed))
    ag = torch.tensor([[float(x) for x in audiogram.split(",")]], dtype=torch.float32)
    out_path = create_comparison_video(
        original_mp4=input,
        output_dir=Path(cfg.paths.outputs) / "video",
        baseline_ckpt=baseline_ckpt,
        conditioned_ckpt=conditioned_ckpt,
        audiogram=ag,
        config_path=config,
    )
    console.log(f"Saved comparison video to {out_path}")


if __name__ == "__main__":
    app()
