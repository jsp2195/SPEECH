from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.profiles import (
    HearingProfile,
    load_profile,
    resolve_profile_input,
    save_profile,
)
from personalized_hearing_enhancement.audiometry.stimuli import generate_tone_probe
from personalized_hearing_enhancement.cli.main import app


def test_tone_probe_is_finite_and_not_clipping() -> None:
    wave = generate_tone_probe(1000.0, amplitude=0.8, duration_s=0.25, sr=16000, ramp_ms=10.0)
    assert wave.ndim == 1
    assert np.isfinite(wave).all()
    assert float(np.max(np.abs(wave))) <= 1.0 + 1e-6


def test_simulated_threshold_estimation_converges() -> None:
    gt = [15, 20, 25, 35, 45, 50, 55, 60]
    _, profile = run_hearing_test(
        AudiometryEngineConfig(step_size_db=8.0, min_step_size_db=2.0, max_trials_per_frequency=16, max_reversals=4),
        mode="simulated",
        ground_truth_audiogram=gt,
        seed=42,
    )
    est = np.array(profile.thresholds_db)
    err = np.abs(est - np.array(gt))
    assert float(np.mean(err)) <= 8.0


def test_profile_roundtrip_and_validation(tmp_path: Path) -> None:
    profile = HearingProfile(
        frequencies=[250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        thresholds_db=[10, 20, 30, 40, 50, 60, 70, 80],
        uncertainty=[2] * 8,
        source="estimated",
        sample_rate=16000,
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.thresholds_db == profile.thresholds_db

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"frequencies": [250], "thresholds_db": [10], "source": "estimated"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(bad)


def test_cli_simulated_estimation_and_show_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    out_profile = tmp_path / "estimated.json"

    result = runner.invoke(
        app,
        [
            "estimate-profile",
            "--mode",
            "simulated",
            "--seed",
            "7",
            "--simulated-audiogram",
            "20,25,30,45,60,65,70,75",
            "--output_profile_json",
            str(out_profile),
        ],
    )
    assert result.exit_code == 0
    assert out_profile.exists()

    result2 = runner.invoke(app, ["show-profile", "--profile_json", str(out_profile)])
    assert result2.exit_code == 0
    assert "Thresholds:" in result2.stdout


def test_profile_json_precedence_over_manual_audiogram(tmp_path: Path) -> None:
    from personalized_hearing_enhancement.audiometry.profiles import resolve_audiogram_tensor

    profile = HearingProfile(
        frequencies=[250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        thresholds_db=[1, 2, 3, 4, 5, 6, 7, 8],
        source="simulated",
    )
    profile_path = tmp_path / "p.json"
    save_profile(profile, profile_path)
    tensor, source = resolve_audiogram_tensor(str(profile_path), "20,25,30,45,60,65,70,75")
    assert source.startswith("profile_json")
    assert torch.allclose(tensor, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32))


def test_resolve_profile_input_prefers_profile_json(tmp_path: Path) -> None:
    profile = HearingProfile(
        frequencies=[250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        thresholds_db=[10, 11, 12, 13, 14, 15, 16, 17],
        source="simulated",
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)

    resolved, source = resolve_profile_input(str(path), "1,1,1,1,1,1,1,1")
    assert source.startswith("profile_json")
    assert resolved.thresholds_db == profile.thresholds_db


def test_profile_tensor_roundtrip_and_manual_path() -> None:
    resolved, source = resolve_profile_input(None, "20,25,30,45,60,65,70,75")
    assert source == "manual"
    tensor = resolved.to_tensor()
    assert tensor.shape == (1, 8)
    assert torch.allclose(tensor, torch.tensor([[20, 25, 30, 45, 60, 65, 70, 75]], dtype=torch.float32))
