from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.audiometry.profiles import HearingProfile, resolve_profile_input
from personalized_hearing_enhancement.evaluation.metrics import bandwise_energy, gain_stats, three_way_user_benefit_metrics
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.calibration_filter import apply_profile_calibration, resolve_device_profile
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.simulation.loudness import clipping_stats, safe_post_amplification
from personalized_hearing_enhancement.utils.audio import load_audio, normalize_audio, save_audio
from personalized_hearing_enhancement.utils.logging_utils import build_logger, log_json
from personalized_hearing_enhancement.utils.plotting import save_spectrogram_plot, save_waveform_plot
from personalized_hearing_enhancement.utils.repro import set_global_seed

app = typer.Typer()


def _load_model(kind: str, ckpt_path: Path, cfg, logger) -> torch.nn.Module:
    kwargs = dict(cfg.model.tasnet)
    model = ConvTasNet(**kwargs) if kind == "baseline" else ConditionedConvTasNet(**kwargs)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
    else:
        logger.warning(f"Checkpoint missing for {kind}: {ckpt_path}. Using random weights.")
    model.eval()
    return model


def _save_hf_plot(metrics: dict[str, dict], output: Path) -> None:
    labels = ["impaired", "calibration", "conditioned"]
    values = [metrics[label]["hf_energy_ratio"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color=["#999999", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("HF energy ratio")
    ax.set_title("High-frequency restoration comparison")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _save_safety_plot(metrics: dict[str, dict], output: Path) -> None:
    labels = ["impaired", "calibration", "conditioned"]
    values = [metrics[label]["safety_clipping_fraction"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color=["#999999", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("Clipping fraction")
    ax.set_title("Safety summary (lower is better)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _human_summary(metrics: dict[str, dict]) -> str:
    ls = metrics["listener_space"]
    comp = metrics["comparison"]
    return (
        "Three-way user-benefit summary\n"
        "1) impaired input: "
        f"listener_si_sdr={ls['impaired']['listener_space_si_sdr']:.3f}, "
        f"listener_spectral_distance={ls['impaired']['listener_space_spectral_distance']:.3f}\n"
        "2) calibration baseline: "
        f"listener_si_sdr={ls['calibration']['listener_space_si_sdr']:.3f}, "
        f"listener_spectral_distance={ls['calibration']['listener_space_spectral_distance']:.3f}\n"
        "3) conditioned ML: "
        f"listener_si_sdr={ls['conditioned']['listener_space_si_sdr']:.3f}, "
        f"listener_spectral_distance={ls['conditioned']['listener_space_spectral_distance']:.3f}\n"
        "4) conditioned-vs-calibration: "
        f"delta_listener_si_sdr={comp['conditioned_vs_calibration_listener_space_delta']['listener_space_si_sdr']:.3f}, "
        f"delta_listener_spectral_distance={comp['conditioned_vs_calibration_listener_space_delta']['listener_space_spectral_distance']:.3f}"
    )


def run_demo_audio(
    input_wav: str,
    config: str,
    baseline_ckpt: str,
    conditioned_ckpt: str,
    audiogram: str | None,
    output_dir: str,
    run_name: str,
    mode: str = "model",
    device_profile: str = "headphones",
    max_gain_db: float = 20.0,
    debug: bool = False,
    profile_json: str | None = None,
    profile: HearingProfile | None = None,
) -> Path:
    cfg = OmegaConf.load(config)
    set_global_seed(int(cfg.seed))
    sr = int(cfg.sample_rate)

    out = Path(output_dir) / run_name
    logger = build_logger(out, name=f"phe_demo_{run_name}")

    if profile is None:
        profile, profile_source = resolve_profile_input(
            profile_json,
            audiogram,
            logger=logger,
            sample_rate=sr,
            default_device_profile=device_profile,
        )
    else:
        profile_source = "provided_profile"

    audiogram_tensor = profile.to_tensor()
    logger.info(f"Using hearing profile source: {profile_source}")

    selected_device_profile = device_profile
    if profile.device_profile and device_profile == "headphones":
        selected_device_profile = profile.device_profile
    profile_name, _, profile_warning = resolve_device_profile(selected_device_profile, debug=debug)
    if profile_warning:
        logger.warning(profile_warning)

    original = normalize_audio(load_audio(input_wav, sr))
    impaired = apply_hearing_loss(original.unsqueeze(0), audiogram_tensor, sr=sr).squeeze(0)

    calibration = apply_profile_calibration(
        original,
        profile,
        sample_rate=sr,
        device_profile=profile_name,
        max_gain_db=max_gain_db,
        chunk_size=512,
    )
    calibration = safe_post_amplification(calibration.unsqueeze(0), reference=original.unsqueeze(0)).squeeze(0)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg, logger)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg, logger)

    with torch.no_grad():
        baseline_enh = baseline(original.unsqueeze(0)).squeeze(0)
        conditioned_enh = conditioned(original.unsqueeze(0), audiogram_tensor).squeeze(0)

    baseline_enh = safe_post_amplification(baseline_enh.unsqueeze(0), reference=original.unsqueeze(0)).squeeze(0)
    conditioned_enh = safe_post_amplification(conditioned_enh.unsqueeze(0), reference=original.unsqueeze(0)).squeeze(0)

    output_wave = conditioned_enh if mode == "model" else calibration

    save_audio(out / "original.wav", original, sr)
    save_audio(out / "impaired.wav", impaired, sr)
    save_audio(out / "clean.wav", original, sr)
    save_audio(out / "degraded.wav", impaired, sr)
    save_audio(out / "baseline.wav", baseline_enh, sr)
    save_audio(out / "calibration.wav", calibration, sr)
    save_audio(out / "conditioned.wav", conditioned_enh, sr)
    save_audio(out / "output.wav", output_wave, sr)

    waves = {
        "original": original,
        "impaired": impaired,
        "calibration-baseline": calibration,
        "conditioned-ml": conditioned_enh,
        "baseline-model": baseline_enh,
    }
    save_waveform_plot(waves, out / "waveforms.png", sr=sr)
    save_spectrogram_plot(waves, out / "spectrograms.png", sr=sr)

    three_way_metrics = three_way_user_benefit_metrics(
        original=original,
        impaired=impaired,
        calibration=calibration,
        conditioned=conditioned_enh,
        audiogram=audiogram_tensor,
        sr=sr,
    )
    three_way_metrics["signal_space"]["baseline_model"] = {
        "signal_space_band_energy": bandwise_energy(baseline_enh, sr=sr),
        "signal_space_gain": gain_stats(original, baseline_enh),
        "signal_space_clipping": clipping_stats(baseline_enh),
    }

    metric_rows = {
        "mode": mode,
        "baseline_role": "calibration",
        "challenger_role": "conditioned_ml",
        "device_profile": profile_name,
        "max_gain_db": max_gain_db,
        "profile": profile.as_metadata(),
        **three_way_metrics,
    }

    _save_hf_plot(metric_rows["hf"], out / "hf_energy_comparison.png")
    _save_safety_plot(metric_rows["safety"], out / "safety_summary.png")

    summary = _human_summary(metric_rows)
    logger.info(summary)
    print(summary)
    log_json(out, "metrics.json", metric_rows)
    return out


@app.command()
def main(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str | None = None,
    output_dir: str = "outputs",
    run_name: str = "demo_audio",
    mode: str = "model",
    device_profile: str = "headphones",
    max_gain_db: float = 20.0,
    debug: bool = False,
    profile_json: str | None = None,
) -> None:
    run_demo_audio(
        input_wav,
        config,
        baseline_ckpt,
        conditioned_ckpt,
        audiogram,
        output_dir,
        run_name,
        mode=mode,
        device_profile=device_profile,
        max_gain_db=max_gain_db,
        debug=debug,
        profile_json=profile_json,
    )


if __name__ == "__main__":
    app()
