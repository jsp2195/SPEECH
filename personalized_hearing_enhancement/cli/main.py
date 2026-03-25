from __future__ import annotations

from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.data.download import download_datasets
from personalized_hearing_enhancement.evaluation.demo_audio import run_demo_audio
from personalized_hearing_enhancement.training.train import run_training
from personalized_hearing_enhancement.utils.audio import load_audio, save_audio
from personalized_hearing_enhancement.utils.logging_utils import build_logger
from personalized_hearing_enhancement.utils.repro import set_global_seed
from personalized_hearing_enhancement.video.video_pipeline import create_comparison_video

app = typer.Typer(help="Audiogram-conditioned speech enhancement system")


def _get_cfg(config: str):
    cfg = OmegaConf.load(config)
    set_global_seed(int(cfg.seed))
    return cfg


def _find_debug_wav(cfg) -> Path:
    candidates = list((Path(cfg.paths.data_cache) / "LibriSpeech" / "dev-clean").rglob("*.flac"))
    if not candidates:
        raise FileNotFoundError("No debug audio source found in LibriSpeech dev-clean.")
    wav = load_audio(candidates[0], sr=int(cfg.sample_rate))
    out = Path(cfg.paths.outputs) / "debug_input.wav"
    save_audio(out, wav, int(cfg.sample_rate))
    return out


@app.command()
def prepare_data(config: str = "personalized_hearing_enhancement/configs/default.yaml") -> None:
    cfg = _get_cfg(config)
    download_datasets(cfg.paths.data_cache)


@app.command()
def train(
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    model_type: str = typer.Option("baseline", help="baseline or conditioned"),
    debug: bool = typer.Option(False, "--debug"),
    overfit_single_batch: bool = typer.Option(False, "--overfit_single_batch"),
    run_name: str = "train",
) -> None:
    _get_cfg(config)
    run_training(config, model_type, debug=debug, overfit_single_batch=overfit_single_batch, run_name=run_name)


@app.command()
def demo_audio(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str = "20,25,30,45,60,65,70,75",
    run_name: str = "demo_audio",
) -> None:
    _get_cfg(config)
    run_demo_audio(input_wav, config, baseline_ckpt, conditioned_ckpt, audiogram, "outputs", run_name)


@app.command()
def process_video(
    input: str = typer.Option(..., "--input", help="Input MP4"),
    audiogram: str = typer.Option("20,25,30,45,60,65,70,75", help="8 comma-separated dB losses"),
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    run_name: str = "video",
) -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / run_name, name=f"phe_video_{run_name}")
    ag = torch.tensor([[float(x) for x in audiogram.split(",")]], dtype=torch.float32)
    out_path = create_comparison_video(
        original_mp4=input,
        output_dir=Path(cfg.paths.outputs) / run_name,
        baseline_ckpt=baseline_ckpt,
        conditioned_ckpt=conditioned_ckpt,
        audiogram=ag,
        config_path=config,
    )
    logger.info(f"Saved comparison video to {out_path}")


@app.command()
def debug(config: str = "personalized_hearing_enhancement/configs/default.yaml") -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / cfg.debug.run_name, name="phe_debug")
    download_datasets(cfg.paths.data_cache)
    run_training(config, "baseline", debug=True, overfit_single_batch=False, run_name=cfg.debug.run_name)
    run_training(config, "conditioned", debug=True, overfit_single_batch=False, run_name=f"{cfg.debug.run_name}_conditioned")
    input_wav = _find_debug_wav(cfg)
    run_demo_audio(
        str(input_wav),
        config,
        "outputs/checkpoints/baseline_best.pt",
        "outputs/checkpoints/conditioned_best.pt",
        "20,25,30,45,60,65,70,75",
        "outputs",
        cfg.debug.run_name,
    )
    logger.info("Debug pipeline complete.")


if __name__ == "__main__":
    app()
