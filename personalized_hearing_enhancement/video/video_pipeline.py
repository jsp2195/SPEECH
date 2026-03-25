from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from personalized_hearing_enhancement.audiometry.profiles import HearingProfile, resolve_profile_input
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.calibration_filter import apply_profile_calibration, resolve_device_profile
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.simulation.loudness import safe_post_amplification
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


def process_audio(
    wav_path: str | Path,
    baseline_ckpt: str | Path,
    conditioned_ckpt: str | Path,
    profile: HearingProfile,
    config_path: str,
    output_dir: str | Path,
    device_profile: str | None = None,
    max_gain_db: float = 20.0,
) -> dict[str, Path]:
    cfg = OmegaConf.load(config_path)
    sr = int(cfg.sample_rate)
    wav = load_audio(wav_path, sr=sr).unsqueeze(0)
    audiogram = profile.to_tensor()
    impaired = apply_hearing_loss(wav, audiogram, sr=sr)

    baseline = _load_model("baseline", Path(baseline_ckpt), cfg)
    conditioned = _load_model("conditioned", Path(conditioned_ckpt), cfg)

    with torch.no_grad():
        b = baseline(wav)
        c = conditioned(wav, audiogram)

    selected_device = device_profile
    if selected_device is None:
        selected_device = profile.get_device_profile("headphones") or "headphones"
    profile_name, _, _ = resolve_device_profile(selected_device)

    calibration = apply_profile_calibration(
        wav.squeeze(0),
        profile,
        sample_rate=sr,
        device_profile=profile_name,
        max_gain_db=max_gain_db,
        chunk_size=512,
    ).unsqueeze(0)

    b = safe_post_amplification(b, reference=wav)
    c = safe_post_amplification(c, reference=wav)
    calibration = safe_post_amplification(calibration, reference=wav)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_p = out_dir / "original.wav"
    impaired_p = out_dir / "impaired.wav"
    baseline_p = out_dir / "baseline.wav"
    cal_p = out_dir / "calibration.wav"
    conditioned_p = out_dir / "conditioned.wav"
    save_audio(original_p, wav.squeeze(0), sr)
    save_audio(out_dir / "clean.wav", wav.squeeze(0), sr)
    save_audio(impaired_p, impaired.squeeze(0), sr)
    save_audio(out_dir / "degraded.wav", impaired.squeeze(0), sr)
    save_audio(baseline_p, b.squeeze(0), sr)
    save_audio(cal_p, calibration.squeeze(0), sr)
    save_audio(conditioned_p, c.squeeze(0), sr)
    return {
        "original": original_p,
        "impaired": impaired_p,
        "baseline": baseline_p,
        "calibration": cal_p,
        "conditioned": conditioned_p,
    }


def _video_with_audio(input_video: Path, audio_path: Path, output_video: Path, label: str) -> None:
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(audio_path),
        "-vf",
        f"drawtext=text='{label}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6",
        "-c:v",
        "libx264",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_video),
    ])


def _create_visual_grid(vids: dict[str, Path], out: Path) -> Path:
    blank = out / "blank.mp4"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=640x360:d=3600", "-an", str(blank)])

    grid_out = out / "comparison_grid_visual_only.mp4"
    filter_graph = (
        "[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[2:v]scale=640:360[v2];"
        "[3:v]scale=640:360[v3];[4:v]scale=640:360[v4];[5:v]scale=640:360[v5];"
        "[v0][v1][v2]hstack=inputs=3[row0];[v3][v4][v5]hstack=inputs=3[row1];"
        "[row0][row1]vstack=inputs=2[v]"
    )
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(vids["original"]),
        "-i",
        str(vids["impaired"]),
        "-i",
        str(vids["calibration"]),
        "-i",
        str(vids["baseline"]),
        "-i",
        str(vids["conditioned"]),
        "-i",
        str(blank),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        str(grid_out),
    ])
    return grid_out


def _create_sequential_comparison(vids: dict[str, Path], out: Path) -> Path:
    list_file = out / "sequential_inputs.txt"
    order = ["original", "impaired", "calibration", "baseline", "conditioned"]
    with list_file.open("w", encoding="utf-8") as f:
        for key in order:
            if key not in vids or not vids[key].exists():
                raise RuntimeError(f"Missing segment video for sequential comparison: {key}")
            f.write(f"file '{vids[key].resolve()}'\n")

    seq_out = out / "comparison_sequential.mp4"
    _run([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(seq_out),
    ])
    return seq_out


def create_comparison_video(
    original_mp4: str | Path,
    output_dir: str | Path,
    baseline_ckpt: str | Path,
    conditioned_ckpt: str | Path,
    audiogram: torch.Tensor | None,
    config_path: str,
    device_profile: str | None = "headphones",
    max_gain_db: float = 20.0,
    profile_json: str | None = None,
    profile: HearingProfile | None = None,
) -> Path:
    ensure_ffmpeg()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(original_mp4)

    if profile is None:
        manual = None if audiogram is None else ",".join(str(float(v)) for v in audiogram.squeeze(0).tolist())
        profile, _ = resolve_profile_input(profile_json, manual, sample_rate=int(OmegaConf.load(config_path).sample_rate))

    wav = extract_audio(source, out / "extracted.wav")
    stems = process_audio(
        wav,
        baseline_ckpt,
        conditioned_ckpt,
        profile,
        config_path,
        out,
        device_profile=device_profile,
        max_gain_db=max_gain_db,
    )

    labels = {
        "original": "1) Original",
        "impaired": "2) Hearing-Impaired (Illustrative)",
        "calibration": "3) Calibration Baseline (DSP)",
        "baseline": "4) Baseline Model",
        "conditioned": "5) Conditioned ML (Profile)",
    }

    vids: dict[str, Path] = {}
    for key, ap in stems.items():
        vp = out / f"{key}.mp4"
        _video_with_audio(source, ap, vp, labels[key])
        vids[key] = vp

    _create_visual_grid(vids, out)
    return _create_sequential_comparison(vids, out)
