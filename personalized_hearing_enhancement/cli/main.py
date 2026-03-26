from __future__ import annotations

from pathlib import Path

import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, SimulatedResponderConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.validation import run_validation_suite, save_validation_summary
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


def _build_audiometry_engine_cfg(
    cfg,
    *,
    estimator_mode: str | None = None,
    lapse_rate: float | None = None,
    guess_rate: float | None = None,
) -> AudiometryEngineConfig:
    audiometry_cfg = cfg.get("audiometry", {}) if hasattr(cfg, "get") else {}
    bayes = audiometry_cfg.get("bayesian", {}) if hasattr(audiometry_cfg, "get") else {}

    return AudiometryEngineConfig(
        sample_rate=int(cfg.sample_rate),
        estimator_mode=(estimator_mode or str(audiometry_cfg.get("estimator_mode", "bayesian"))),
        start_amplitude_db_hl=float(audiometry_cfg.get("start_amplitude_db_hl", 40.0)),
        step_size_db=float(audiometry_cfg.get("step_size_db", 10.0)),
        min_step_size_db=float(audiometry_cfg.get("min_step_size_db", 2.0)),
        max_trials_per_frequency=int(audiometry_cfg.get("max_trials_per_frequency", 18)),
        max_reversals=int(audiometry_cfg.get("max_reversals", 4)),
        threshold_min_db_hl=float(bayes.get("threshold_min_db_hl", 0.0)),
        threshold_max_db_hl=float(bayes.get("threshold_max_db_hl", 100.0)),
        threshold_step_db=float(bayes.get("threshold_step_db", 2.0)),
        psychometric_slope=float(bayes.get("psychometric_slope", 0.35)),
        lapse_rate=float(bayes.get("lapse_rate", 0.0) if lapse_rate is None else lapse_rate),
        guess_rate=float(bayes.get("guess_rate", 0.5) if guess_rate is None else guess_rate),
        candidate_amplitudes_db_hl=[float(v) for v in bayes.get("candidate_amplitudes_db_hl", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])],
        variance_stop_threshold=float(bayes.get("variance_stop_threshold", 9.0)),
        entropy_stop_threshold=float(bayes.get("entropy_stop_threshold", 1.4)),
        min_trials_per_frequency=int(bayes.get("min_trials_per_frequency", 6)),
        low_reliability_threshold=float(bayes.get("low_reliability_threshold", 0.45)),
    )


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
    audiometry_mode: str = typer.Option("bayesian", "--audiometry_mode", help="bayesian or staircase"),
    lapse_rate: float | None = typer.Option(None, "--lapse_rate"),
    guess_rate: float | None = typer.Option(None, "--guess_rate"),
    response_model: str = typer.Option("clean_logistic", "--response_model", help="clean_logistic or lapse_logistic"),
    simulate_fatigue: bool = typer.Option(False, "--simulate_fatigue"),
    inconsistency_rate: float = typer.Option(0.0, "--inconsistency_rate"),
    fatigue_lapse_increment: float = typer.Option(0.0, "--fatigue_lapse_increment"),
) -> None:
    cfg = _get_cfg(config)
    engine_cfg = _build_audiometry_engine_cfg(cfg, estimator_mode=audiometry_mode, lapse_rate=lapse_rate, guess_rate=guess_rate)
    gt = parse_manual_audiogram(simulated_audiogram) if mode == "simulated" else None
    simulated_responder = SimulatedResponderConfig(
        psychometric_slope=engine_cfg.psychometric_slope,
        lapse_rate=engine_cfg.lapse_rate,
        guess_rate=engine_cfg.guess_rate,
        response_model=response_model,
        simulate_fatigue=simulate_fatigue,
        inconsistency_rate=inconsistency_rate,
        fatigue_lapse_increment=fatigue_lapse_increment,
    )
    run_hearing_test(
        engine_cfg,
        mode=mode,
        ground_truth_audiogram=gt,
        seed=seed,
        save_progress_path=save_progress_path,
        simulated_responder=simulated_responder if mode == "simulated" else None,
    )


@app.command("estimate-profile")
def estimate_profile(
    output_profile_json: str = typer.Option("outputs/estimated_profile.json", "--output_profile_json"),
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    mode: str = typer.Option("interactive", help="interactive or simulated"),
    simulated_audiogram: str = typer.Option("20,25,30,45,60,65,70,75", help="8 comma-separated dB HL thresholds"),
    seed: int = typer.Option(0, "--seed"),
    notes: str = typer.Option("", "--notes"),
    audiometry_mode: str = typer.Option("bayesian", "--audiometry_mode", help="bayesian or staircase"),
    lapse_rate: float | None = typer.Option(None, "--lapse_rate"),
    guess_rate: float | None = typer.Option(None, "--guess_rate"),
    response_model: str = typer.Option("clean_logistic", "--response_model", help="clean_logistic or lapse_logistic"),
    simulate_fatigue: bool = typer.Option(False, "--simulate_fatigue"),
    inconsistency_rate: float = typer.Option(0.0, "--inconsistency_rate"),
    fatigue_lapse_increment: float = typer.Option(0.0, "--fatigue_lapse_increment"),
) -> None:
    cfg = _get_cfg(config)
    engine_cfg = _build_audiometry_engine_cfg(cfg, estimator_mode=audiometry_mode, lapse_rate=lapse_rate, guess_rate=guess_rate)
    gt = parse_manual_audiogram(simulated_audiogram) if mode == "simulated" else None
    simulated_responder = SimulatedResponderConfig(
        psychometric_slope=engine_cfg.psychometric_slope,
        lapse_rate=engine_cfg.lapse_rate,
        guess_rate=engine_cfg.guess_rate,
        response_model=response_model,
        simulate_fatigue=simulate_fatigue,
        inconsistency_rate=inconsistency_rate,
        fatigue_lapse_increment=fatigue_lapse_increment,
    )
    _, profile = run_hearing_test(
        engine_cfg,
        mode=mode,
        ground_truth_audiogram=gt,
        seed=seed,
        simulated_responder=simulated_responder if mode == "simulated" else None,
    )
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


@app.command("validate-audiometry")
def validate_audiometry(
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    runs_per_profile: int = typer.Option(5, "--runs_per_profile"),
    psychometric_slope: float = typer.Option(0.35, "--psychometric_slope"),
    jitter_std: float = typer.Option(1.5, "--jitter_std"),
    seed: int = typer.Option(0, "--seed"),
    output_json: str = typer.Option("outputs/audiometry_validation_summary.json", "--output_json"),
    audiometry_mode: str = typer.Option("bayesian", "--audiometry_mode", help="bayesian or staircase"),
    include_staircase_baseline: bool = typer.Option(True, "--include_staircase_baseline"),
    lapse_rate: float | None = typer.Option(None, "--lapse_rate"),
    guess_rate: float | None = typer.Option(None, "--guess_rate"),
) -> None:
    cfg = _get_cfg(config)
    summary = run_validation_suite(
        runs_per_profile=runs_per_profile,
        slope=psychometric_slope,
        base_seed=seed,
        jitter_std=jitter_std,
        engine_cfg=_build_audiometry_engine_cfg(cfg, estimator_mode=audiometry_mode, lapse_rate=lapse_rate, guess_rate=guess_rate),
        include_staircase_baseline=include_staircase_baseline,
    )
    saved = save_validation_summary(summary, output_json)
    print(f"Saved audiometry validation summary: {saved}")
    print(
        f"Mean MAE: {summary['mean_mae']:.2f} dB | Mean trials/profile: {summary['mean_trials_per_profile']:.1f} "
        f"| Mean reliability: {summary.get('mean_reliability_score', float('nan')):.3f}"
    )


@app.command()
def debug(config: str = "personalized_hearing_enhancement/configs/default.yaml") -> None:
    cfg = _get_cfg(config)
    logger = build_logger(Path(cfg.paths.outputs) / cfg.debug.run_name, name="phe_debug")
    download_datasets(cfg.paths.data_cache)
    run_training(config, "baseline", debug=True, overfit_single_batch=False, run_name=cfg.debug.run_name)
    run_training(config, "conditioned", debug=True, overfit_single_batch=False, run_name=f"{cfg.debug.run_name}_conditioned")
    input_wav = _find_debug_wav(cfg)
    _, profile = run_hearing_test(
        _build_audiometry_engine_cfg(cfg),
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
