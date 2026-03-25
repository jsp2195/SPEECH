from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from rich.console import Console

from personalized_hearing_enhancement.evaluation.metrics import si_sdr
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.utils.audio import load_audio, normalize_audio, save_audio

console = Console()
app = typer.Typer()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_model(kind: str, ckpt_path: Path, cfg) -> torch.nn.Module:
    kwargs = dict(cfg.model.tasnet)
    model = ConvTasNet(**kwargs) if kind == "baseline" else ConditionedConvTasNet(**kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


@app.command()
def main(
    input_wav: str,
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    baseline_ckpt: str = "outputs/checkpoints/baseline_best.pt",
    conditioned_ckpt: str = "outputs/checkpoints/conditioned_best.pt",
    audiogram: str = "20,25,30,45,60,65,70,75",
    output_dir: str = "outputs/demo_audio",
) -> None:
    cfg = OmegaConf.load(config)
    set_seed(int(cfg.seed))
    sr = int(cfg.sample_rate)

    clean = normalize_audio(load_audio(input_wav, sr))
    ag = torch.tensor([float(x) for x in audiogram.split(",")], dtype=torch.float32).unsqueeze(0)
    degraded = apply_hearing_loss(clean.unsqueeze(0), ag, sr=sr).squeeze(0)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg)

    with torch.no_grad():
        baseline_enh = baseline(degraded.unsqueeze(0)).squeeze(0)
        conditioned_enh = conditioned(degraded.unsqueeze(0), ag).squeeze(0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_audio(out / "clean.wav", clean, sr)
    save_audio(out / "degraded.wav", degraded, sr)
    save_audio(out / "baseline_enhanced.wav", baseline_enh, sr)
    save_audio(out / "personalized_enhanced.wav", conditioned_enh, sr)

    table = {
        "degraded_si_sdr": si_sdr(clean, degraded).mean().item(),
        "baseline_si_sdr": si_sdr(clean, baseline_enh).mean().item(),
        "personalized_si_sdr": si_sdr(clean, conditioned_enh).mean().item(),
    }
    console.log(table)


if __name__ == "__main__":
    app()
