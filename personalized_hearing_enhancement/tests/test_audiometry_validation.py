from __future__ import annotations

import math

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, logistic_hear_probability
from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ
from personalized_hearing_enhancement.audiometry.validation import PROFILE_LIBRARY, SCENARIOS, generate_synthetic_profile, run_single_validation, run_validation_suite


def test_synthetic_profile_generation_shape_and_range() -> None:
    for profile_type in PROFILE_LIBRARY:
        profile = generate_synthetic_profile(profile_type, jitter_std=0.5, seed=1)
        assert len(profile) == len(STANDARD_FREQS_HZ)
        assert all(0.0 <= x <= 100.0 for x in profile)


def test_logistic_response_probability_monotonicity() -> None:
    threshold = 40.0
    below = logistic_hear_probability(25.0, threshold, slope=0.4)
    at = logistic_hear_probability(40.0, threshold, slope=0.4)
    above = logistic_hear_probability(55.0, threshold, slope=0.4)
    assert below < at < above


def test_validation_loop_runs_end_to_end() -> None:
    result = run_single_validation(
        "normal",
        engine_cfg=AudiometryEngineConfig(infer_device_gain=True),
        true_device_gain_db=6.0,
        slope=0.35,
        seed=3,
        jitter_std=0.0,
    )
    assert len(result.estimated_thresholds) == len(STANDARD_FREQS_HZ)
    assert result.total_trials > 0
    assert math.isfinite(result.mean_abs_error)
    assert math.isfinite(result.device_gain_abs_error)


def test_validation_summary_shapes_and_finiteness() -> None:
    summary = run_validation_suite(
        runs_per_profile=2,
        slope=0.35,
        base_seed=5,
        jitter_std=0.5,
        engine_cfg=AudiometryEngineConfig(infer_device_gain=True),
    )
    assert math.isfinite(summary["mean_mae"])
    assert math.isfinite(summary["mean_gain_abs_error"])
    assert len(summary["mae_by_frequency"]) == len(STANDARD_FREQS_HZ)
    assert len(summary["runs"]) == len(SCENARIOS) * 2
