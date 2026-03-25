from __future__ import annotations

import math

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, logistic_hear_probability
from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ
from personalized_hearing_enhancement.audiometry.validation import (
    PROFILE_LIBRARY,
    generate_synthetic_profile,
    run_single_validation,
    run_validation_suite,
)


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
    assert 0.0 < below < 1.0
    assert 0.0 < above < 1.0


def test_validation_loop_runs_end_to_end() -> None:
    result = run_single_validation(
        "mild_loss",
        engine_cfg=AudiometryEngineConfig(max_trials_per_frequency=12, max_reversals=3, step_size_db=8.0),
        slope=0.35,
        seed=3,
        jitter_std=0.0,
    )
    assert len(result.estimated_thresholds) == len(STANDARD_FREQS_HZ)
    assert result.total_trials > 0
    assert math.isfinite(result.mean_abs_error)


def test_recovery_reasonable_for_simple_case() -> None:
    result = run_single_validation(
        "normal",
        engine_cfg=AudiometryEngineConfig(max_trials_per_frequency=14, max_reversals=4, step_size_db=6.0),
        slope=0.45,
        seed=7,
        jitter_std=0.0,
    )
    assert result.mean_abs_error < 20.0


def test_validation_summary_shapes_and_finiteness() -> None:
    summary = run_validation_suite(
        runs_per_profile=2,
        slope=0.35,
        base_seed=5,
        jitter_std=0.5,
        engine_cfg=AudiometryEngineConfig(max_trials_per_frequency=10, max_reversals=3),
    )
    assert math.isfinite(summary["mean_mae"])
    assert math.isfinite(summary["median_mae"])
    assert len(summary["mae_by_frequency"]) == len(STANDARD_FREQS_HZ)
    assert len(summary["error_distribution"]) == len(PROFILE_LIBRARY) * 2
    assert math.isfinite(summary["mean_trials_per_profile"])
