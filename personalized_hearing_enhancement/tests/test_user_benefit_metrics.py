from __future__ import annotations

import torch

from personalized_hearing_enhancement.evaluation.metrics import three_way_user_benefit_metrics


def test_three_way_user_benefit_metrics_contains_required_sections() -> None:
    sr = 16000
    t = torch.linspace(0, 0.5, int(0.5 * sr))
    original = torch.sin(2 * torch.pi * 440 * t)
    impaired = original * 0.6
    calibration = original * 0.9
    conditioned = original * 0.95
    audiogram = torch.tensor([[20, 25, 30, 45, 60, 65, 70, 75]], dtype=torch.float32)

    metrics = three_way_user_benefit_metrics(original, impaired, calibration, conditioned, audiogram, sr=sr)

    assert set(metrics.keys()) == {"signal_space", "listener_space", "hf", "safety", "comparison"}
    assert "conditioned_vs_calibration_listener_space_delta" in metrics["comparison"]
    assert "conditioned_vs_calibration_hf_delta" in metrics["comparison"]
    assert "conditioned_vs_calibration_safety_delta" in metrics["comparison"]
