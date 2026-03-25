from __future__ import annotations

from pathlib import Path

import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.profiles import (
    HearingProfile,
    load_profile,
    parse_manual_audiogram,
    print_profile_summary,
    resolve_profile_input,
    save_profile,
    save_profile_plot,
)
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


@app.command("run-hearing-test")
def run_hearing_test_cmd(
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    mode: str = typer.Option("interactive", help="interactive or simulated"),
    simulated_audiogram: str = typer.Option("20,25,30,45,60,65,70,75", help="8 comma-separated dB HL thresholds"),
    seed: int = typer.Option(0, "--seed"),
    save_progress_path: str | None = typer.Option(None, "--save_progress_path"),
) -> None:
    cfg = _get_cfg(config)
    engine_cfg = AudiometryEngineConfig(sample_rate=int(cfg.sample_rate))
    gt = parse_manual_audiogram(simulated_audiogram) if mode == "simulated" else None
    run_hearing_test(engine_cfg, mode=mode, ground_truth_audiogram=gt, seed=seed, save_progress_path=save_progress_path)


@app.command("estimate-profile")
def estimate_profile(
    output_profile_json: str = typer.Option("outputs/estimated_profile.json", "--output_profile_json"),
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    mode: str = typer.Option("interactive", help="interactive or simulated"),
    simulated_audiogram: str = typer.Option("20,25,30,45,60,65,70,75", help="8 comma-separated dB HL thresholds"),
    seed: int = typer.Option(0, "--seed"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    cfg = _get_cfg(config)
    engine_cfg = AudiometryEngineConfig(sample_rate=int(cfg.sample_rate))
    gt = parse_manual_audiogram(simulated_audiogram) if mode == "simulated" else None
    _, profile = run_hearing_test(engine_cfg, mode=mode, ground_truth_audiogram=gt, seed=seed)
    profile.notes = notes
    saved = save_profile(profile, output_profile_json)
    print(f"Saved hearing profile: {saved}")


@app.command("show-profile")
def show_profile(
    profile_json: str = typer.Option(..., "--profile_json"),
    save_plot_path: str | None = typer.Option(None, "--save_plot_path"),
) -> None:
    profile = load_profile(profile_json)
    print(print_profile_summary(profile))
    if save_plot_path:
        out = save_profile_plot(profile, save_plot_path)
        print(f"Saved profile plot: {out}")


@app.command()
def demo_audio(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str | None = typer.Option(None, help="8 comma-separated dB losses (manual fallback)"),
    profile_json: str | None = typer.Option(None, "--profile_json", help="Path to saved hearing profile JSON"),
    run_name: str = "demo_audio",
    mode: str = typer.Option("model", "--mode", help="model or calibration"),
    device_profile: str = typer.Option("headphones", "--device_profile", help="earbuds|headphones|airpods|overear"),
    max_gain_db: float = typer.Option(20.0, "--max_gain_db"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / run_name, name=f"phe_demo_cli_{run_name}")
    profile, source = resolve_profile_input(
        profile_json,
        audiogram,
        logger=logger,
        sample_rate=int(cfg.sample_rate),
        default_device_profile=device_profile,
    )
    logger.info(f"Resolved profile input path: {source}")
    logger.info(f"Device profile source: {profile.device_profile or 'cli/default'}")
    run_demo_audio(
        input_wav,
        config,
        baseline_ckpt,
        conditioned_ckpt,
        audiogram,
        str(cfg.paths.outputs),
        run_name,
        mode=mode,
        device_profile=device_profile,
        max_gain_db=max_gain_db,
        debug=debug,
        profile_json=profile_json,
        profile=profile,
    )


@app.command()
def process_video(
    input: str = typer.Option(..., "--input", help="Input MP4"),
    audiogram: str | None = typer.Option(None, help="8 comma-separated dB losses (manual fallback)"),
    profile_json: str | None = typer.Option(None, "--profile_json", help="Path to saved hearing profile JSON"),
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    run_name: str = "video",
    device_profile: str = typer.Option("headphones", "--device_profile", help="earbuds|headphones|airpods|overear"),
    max_gain_db: float = typer.Option(20.0, "--max_gain_db"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / run_name, name=f"phe_video_{run_name}")
    profile, source = resolve_profile_input(
        profile_json,
        audiogram,
        logger=logger,
        sample_rate=int(cfg.sample_rate),
        default_device_profile=device_profile,
    )
    logger.info(f"Resolved profile input path: {source}")
    logger.info(f"Device profile source: {profile.device_profile or 'cli/default'}")

    out_path = create_comparison_video(
        original_mp4=input,
        output_dir=Path(cfg.paths.outputs) / run_name,
        baseline_ckpt=baseline_ckpt,
        conditioned_ckpt=conditioned_ckpt,
        audiogram=profile.to_tensor(),
        config_path=config,
        device_profile=device_profile,
        max_gain_db=max_gain_db,
        profile_json=profile_json,
        profile=profile,
    )
    if debug:
        logger.info(f"Debug mode enabled; output={out_path}")
    logger.info(f"Saved comparison video to {out_path}")


@app.command()
def debug(config: str = "personalized_hearing_enhancement/configs/default.yaml") -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / cfg.debug.run_name, name="phe_debug")
    download_datasets(cfg.paths.data_cache)
    run_training(config, "baseline", debug=True, overfit_single_batch=False, run_name=cfg.debug.run_name)
    run_training(config, "conditioned", debug=True, overfit_single_batch=False, run_name=f"{cfg.debug.run_name}_conditioned")
    input_wav = _find_debug_wav(cfg)
    _, profile = run_hearing_test(
        AudiometryEngineConfig(sample_rate=int(cfg.sample_rate)),
        mode="simulated",
        ground_truth_audiogram=[20, 25, 30, 45, 60, 65, 70, 75],
        seed=123,
    )
    sample_profile = HearingProfile(
        frequencies=profile.frequencies,
        thresholds_db=profile.thresholds_db,
        uncertainty=profile.uncertainty,
        source="simulated",
        sample_rate=int(cfg.sample_rate),
        notes="Debug synthetic profile",
    )
    profile_path = save_profile(sample_profile, Path(cfg.paths.outputs) / cfg.debug.run_name / "sample_profile.json")
    run_demo_audio(
        str(input_wav),
        config,
        "outputs/checkpoints/baseline_best.pt",
        "outputs/checkpoints/conditioned_best.pt",
        None,
        "outputs",
        cfg.debug.run_name,
        mode="model",
        device_profile="headphones",
        max_gain_db=20.0,
        debug=True,
        profile_json=str(profile_path),
        profile=sample_profile,
    )
    logger.info("Debug pipeline complete.")


if __name__ == "__main__":
    app()
