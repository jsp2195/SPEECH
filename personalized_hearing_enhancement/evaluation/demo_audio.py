from __future__ import annotations

from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.evaluation.metrics import (
    bandwise_energy,
    gain_stats,
    high_frequency_energy_ratio,
    intelligibility_proxy,
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
) -> Path:
    cfg = OmegaConf.load(config)
    set_global_seed(int(cfg.seed))
    sr = int(cfg.sample_rate)

    out = Path(output_dir) / run_name
    logger = build_logger(out, name=f"phe_demo_{run_name}")

    clean = normalize_audio(load_audio(input_wav, sr))
    ag = torch.tensor([float(x) for x in audiogram.split(",")], dtype=torch.float32).unsqueeze(0)
    degraded = apply_hearing_loss(clean.unsqueeze(0), ag, sr=sr).squeeze(0)

    profile_name, _, profile_warning = resolve_device_profile(device_profile, debug=debug)
    if profile_warning:
        logger.warning(profile_warning)

    calibration = apply_calibration_filter(
        degraded,
        ag,
        sample_rate=sr,
        device_profile=profile_name,
        max_gain_db=max_gain_db,
        chunk_size=512,
    )
    calibration = safe_post_amplification(calibration.unsqueeze(0), reference=degraded.unsqueeze(0)).squeeze(0)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg, logger)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg, logger)

    with torch.no_grad():
        baseline_enh = baseline(degraded.unsqueeze(0)).squeeze(0)
        conditioned_enh = conditioned(degraded.unsqueeze(0), ag).squeeze(0)

    baseline_enh = safe_post_amplification(baseline_enh.unsqueeze(0), reference=degraded.unsqueeze(0)).squeeze(0)
    conditioned_enh = safe_post_amplification(conditioned_enh.unsqueeze(0), reference=degraded.unsqueeze(0)).squeeze(0)

    output_wave = conditioned_enh if mode == "model" else calibration

    save_audio(out / "clean.wav", clean, sr)
    save_audio(out / "degraded.wav", degraded, sr)
    save_audio(out / "baseline.wav", baseline_enh, sr)
    save_audio(out / "calibration.wav", calibration, sr)
    save_audio(out / "conditioned.wav", conditioned_enh, sr)
    save_audio(out / "output.wav", output_wave, sr)

    waves = {
        "original": clean,
        "hearing-impaired": degraded,
        "baseline": baseline_enh,
        "calibration-filter": calibration,
        "ml-model": conditioned_enh,
    }
    save_waveform_plot(waves, out / "waveforms.png", sr=sr)
    save_spectrogram_plot(waves, out / "spectrograms.png", sr=sr)

    metric_rows: dict[str, float | dict[str, float] | str] = {
        "mode": mode,
        "device_profile": profile_name,
        "max_gain_db": max_gain_db,
        "degraded_si_sdr": si_sdr(clean, degraded).mean().item(),
        "baseline_si_sdr": si_sdr(clean, baseline_enh).mean().item(),
        "calibration_si_sdr": si_sdr(clean, calibration).mean().item(),
        "personalized_si_sdr": si_sdr(clean, conditioned_enh).mean().item(),
        "degraded_hf_recovery": high_frequency_energy_ratio(clean, degraded, sr=sr),
        "calibration_hf_recovery": high_frequency_energy_ratio(clean, calibration, sr=sr),
        "conditioned_hf_recovery": high_frequency_energy_ratio(clean, conditioned_enh, sr=sr),
        "degraded_intelligibility": intelligibility_proxy(clean, degraded),
        "calibration_intelligibility": intelligibility_proxy(clean, calibration),
        "conditioned_intelligibility": intelligibility_proxy(clean, conditioned_enh),
        "baseline_band_energy": bandwise_energy(baseline_enh, sr=sr),
        "calibration_band_energy": bandwise_energy(calibration, sr=sr),
        "conditioned_band_energy": bandwise_energy(conditioned_enh, sr=sr),
        "calibration_gain": gain_stats(degraded, calibration),
        "conditioned_gain": gain_stats(degraded, conditioned_enh),
        "calibration_clipping": clipping_stats(calibration),
        "conditioned_clipping": clipping_stats(conditioned_enh),
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
    )


if __name__ == "__main__":
    app()
