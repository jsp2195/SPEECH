from __future__ import annotations

from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from personalized_hearing_enhancement.evaluation.metrics import si_sdr
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
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
) -> Path:
    cfg = OmegaConf.load(config)
    set_global_seed(int(cfg.seed))
    sr = int(cfg.sample_rate)

    out = Path(output_dir) / run_name
    logger = build_logger(out, name=f"phe_demo_{run_name}")

    clean = normalize_audio(load_audio(input_wav, sr))
    ag = torch.tensor([float(x) for x in audiogram.split(",")], dtype=torch.float32).unsqueeze(0)
    degraded = apply_hearing_loss(clean.unsqueeze(0), ag, sr=sr).squeeze(0)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg, logger)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg, logger)

    with torch.no_grad():
        baseline_enh = baseline(degraded.unsqueeze(0)).squeeze(0)
        conditioned_enh = conditioned(degraded.unsqueeze(0), ag).squeeze(0)

    save_audio(out / "clean.wav", clean, sr)
    save_audio(out / "degraded.wav", degraded, sr)
    save_audio(out / "baseline.wav", baseline_enh, sr)
    save_audio(out / "conditioned.wav", conditioned_enh, sr)

    waves = {
        "clean": clean,
        "degraded": degraded,
        "baseline": baseline_enh,
        "conditioned": conditioned_enh,
    }
    save_waveform_plot(waves, out / "waveforms.png", sr=sr)
    save_spectrogram_plot(waves, out / "spectrograms.png", sr=sr)

    table = {
        "degraded_si_sdr": si_sdr(clean, degraded).mean().item(),
        "baseline_si_sdr": si_sdr(clean, baseline_enh).mean().item(),
        "personalized_si_sdr": si_sdr(clean, conditioned_enh).mean().item(),
        "audiogram": ag.squeeze(0).tolist(),
    }
    logger.info(f"Metrics: {table}")
    log_json(out, "metrics.json", table)
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
) -> None:
    run_demo_audio(input_wav, config, baseline_ckpt, conditioned_ckpt, audiogram, output_dir, run_name)


if __name__ == "__main__":
    app()
