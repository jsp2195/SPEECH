from __future__ import annotations

from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.audiometry.profiles import resolve_audiogram_tensor
from personalized_hearing_enhancement.evaluation.metrics import (
    bandwise_energy,
    gain_stats,
    high_frequency_energy_ratio,
    intelligibility_proxy,
    listener_space_metrics,
    log_spectral_distance,
    si_sdr,
)
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.calibration_filter import apply_calibration_filter, resolve_device_profile
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


def run_demo_audio(
    input_wav: str,
    config: str,
    baseline_ckpt: str,
    conditioned_ckpt: str,
    audiogram: str,
    output_dir: str,
    run_name: str,
    mode: str = "model",
    device_profile: str = "headphones",
    max_gain_db: float = 20.0,
    debug: bool = False,
    profile_json: str | None = None,
) -> Path:
    cfg = OmegaConf.load(config)
    set_global_seed(int(cfg.seed))
    sr = int(cfg.sample_rate)

    out = Path(output_dir) / run_name
    logger = build_logger(out, name=f"phe_demo_{run_name}")

    original = normalize_audio(load_audio(input_wav, sr))
    ag, audiogram_source = resolve_audiogram_tensor(profile_json, audiogram, logger=logger)
    logger.info(f"Using audiogram source: {audiogram_source}")
    impaired = apply_hearing_loss(original.unsqueeze(0), ag, sr=sr).squeeze(0)

    profile_name, _, profile_warning = resolve_device_profile(device_profile, debug=debug)
    if profile_warning:
        logger.warning(profile_warning)

    calibration = apply_calibration_filter(
        original,
        ag,
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
        conditioned_enh = conditioned(original.unsqueeze(0), ag).squeeze(0)

    baseline_enh = safe_post_amplification(baseline_enh.unsqueeze(0), reference=original.unsqueeze(0)).squeeze(0)
    conditioned_enh = safe_post_amplification(conditioned_enh.unsqueeze(0), reference=original.unsqueeze(0)).squeeze(0)

    output_wave = conditioned_enh if mode == "model" else calibration

    save_audio(out / "original.wav", original, sr)
    save_audio(out / "impaired.wav", impaired, sr)
    # Backward-compatible file aliases
    save_audio(out / "clean.wav", original, sr)
    save_audio(out / "degraded.wav", impaired, sr)
    save_audio(out / "baseline.wav", baseline_enh, sr)
    save_audio(out / "calibration.wav", calibration, sr)
    save_audio(out / "conditioned.wav", conditioned_enh, sr)
    save_audio(out / "output.wav", output_wave, sr)

    waves = {
        "original": original,
        "hearing-impaired": impaired,
        "baseline": baseline_enh,
        "calibration-filter": calibration,
        "ml-model": conditioned_enh,
    }
    save_waveform_plot(waves, out / "waveforms.png", sr=sr)
    save_spectrogram_plot(waves, out / "spectrograms.png", sr=sr)

    signal_space_metrics: dict[str, float | dict[str, float] | str] = {
        "mode": mode,
        "device_profile": profile_name,
        "max_gain_db": max_gain_db,
        "signal_space_impaired_si_sdr": si_sdr(original, impaired).mean().item(),
        "signal_space_baseline_si_sdr": si_sdr(original, baseline_enh).mean().item(),
        "signal_space_calibration_si_sdr": si_sdr(original, calibration).mean().item(),
        "signal_space_conditioned_si_sdr": si_sdr(original, conditioned_enh).mean().item(),
        "signal_space_impaired_spectral_distance": log_spectral_distance(original, impaired),
        "signal_space_baseline_spectral_distance": log_spectral_distance(original, baseline_enh),
        "signal_space_calibration_spectral_distance": log_spectral_distance(original, calibration),
        "signal_space_conditioned_spectral_distance": log_spectral_distance(original, conditioned_enh),
        "signal_space_impaired_hf_recovery": high_frequency_energy_ratio(original, impaired, sr=sr),
        "signal_space_calibration_hf_recovery": high_frequency_energy_ratio(original, calibration, sr=sr),
        "signal_space_conditioned_hf_recovery": high_frequency_energy_ratio(original, conditioned_enh, sr=sr),
        "signal_space_impaired_intelligibility": intelligibility_proxy(original, impaired),
        "signal_space_calibration_intelligibility": intelligibility_proxy(original, calibration),
        "signal_space_conditioned_intelligibility": intelligibility_proxy(original, conditioned_enh),
        "baseline_band_energy": bandwise_energy(baseline_enh, sr=sr),
        "calibration_band_energy": bandwise_energy(calibration, sr=sr),
        "conditioned_band_energy": bandwise_energy(conditioned_enh, sr=sr),
        "calibration_gain": gain_stats(original, calibration),
        "conditioned_gain": gain_stats(original, conditioned_enh),
        "calibration_clipping": clipping_stats(calibration),
        "conditioned_clipping": clipping_stats(conditioned_enh),
    }

    listener_space = {
        "impaired_input": listener_space_metrics(original, impaired, ag, sr=sr, output_already_impaired=True),
        "baseline": listener_space_metrics(original, baseline_enh, ag, sr=sr),
        "calibration": listener_space_metrics(original, calibration, ag, sr=sr),
        "conditioned": listener_space_metrics(original, conditioned_enh, ag, sr=sr),
    }

    metric_rows: dict[str, float | dict[str, float] | str | list[float]] = {
        "signal_space": signal_space_metrics,
        "listener_space": listener_space,
        "audiogram": ag.squeeze(0).tolist(),
    }
    logger.info(f"Metrics: {metric_rows}")
    log_json(out, "metrics.json", metric_rows)
    return out


@app.command()
def main(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str = "20,25,30,45,60,65,70,75",
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
