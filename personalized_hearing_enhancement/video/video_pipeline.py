from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.utils.audio import load_audio, save_audio


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install with: Ubuntu `sudo apt-get install ffmpeg`, "
            "macOS `brew install ffmpeg`, Windows `choco install ffmpeg`."
        )


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def extract_audio(video_path: str | Path, wav_path: str | Path, sr: int = 16000) -> Path:
    ensure_ffmpeg()
    wav = Path(wav_path)
    wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), str(wav)])
    return wav


def _load_model(kind: str, ckpt: Path, cfg) -> torch.nn.Module:
    kwargs = dict(cfg.model.tasnet)
    model = ConvTasNet(**kwargs) if kind == "baseline" else ConditionedConvTasNet(**kwargs)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
    model.eval()
    return model


def process_audio(wav_path: str | Path, baseline_ckpt: str | Path, conditioned_ckpt: str | Path, audiogram: torch.Tensor, config_path: str, output_dir: str | Path) -> dict[str, Path]:
    cfg = OmegaConf.load(config_path)
    sr = int(cfg.sample_rate)
    wav = load_audio(wav_path, sr=sr).unsqueeze(0)
    degraded = apply_hearing_loss(wav, audiogram, sr=sr)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg)

    with torch.no_grad():
        b = baseline(degraded)
        c = conditioned(degraded, audiogram)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_p = out_dir / "clean.wav"
    deg_p = out_dir / "degraded.wav"
    b_p = out_dir / "baseline.wav"
    c_p = out_dir / "conditioned.wav"
    save_audio(clean_p, wav.squeeze(0), sr)
    save_audio(deg_p, degraded.squeeze(0), sr)
    save_audio(b_p, b.squeeze(0), sr)
    save_audio(c_p, c.squeeze(0), sr)
    return {"original": clean_p, "impaired": deg_p, "baseline": b_p, "conditioned": c_p}


def _video_with_audio(input_video: Path, audio_path: Path, output_video: Path) -> None:
    _run([
        "ffmpeg", "-y", "-i", str(input_video), "-i", str(audio_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_video)
    ])


def create_comparison_video(original_mp4: str | Path, output_dir: str | Path, baseline_ckpt: str | Path, conditioned_ckpt: str | Path, audiogram: torch.Tensor, config_path: str) -> Path:
    ensure_ffmpeg()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(original_mp4)
    wav = extract_audio(source, out / "extracted.wav")
    stems = process_audio(wav, baseline_ckpt, conditioned_ckpt, audiogram, config_path, out)

    vids: dict[str, Path] = {}
    for key, ap in stems.items():
        vp = out / f"{key}.mp4"
        _video_with_audio(source, ap, vp)
        vids[key] = vp

    grid_out = out / "comparison_grid.mp4"
    filter_graph = (
        "[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[2:v]scale=640:360[v2];[3:v]scale=640:360[v3];"
        "[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]"
    )
    _run([
        "ffmpeg", "-y", "-i", str(vids["original"]), "-i", str(vids["impaired"]), "-i", str(vids["baseline"]), "-i", str(vids["conditioned"]),
        "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a:0", "-shortest", str(grid_out)
    ])
    return grid_out
