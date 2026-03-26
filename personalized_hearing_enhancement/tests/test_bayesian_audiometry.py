from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.inference import BayesianConfig, BayesianThresholdEstimator
from personalized_hearing_enhancement.audiometry.profiles import load_profile, save_profile
from personalized_hearing_enhancement.audiometry.session import AudiometrySession
from personalized_hearing_enhancement.audiometry.validation import run_single_validation


def test_joint_state_initialization_shapes_and_finiteness() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.1, guess_rate=0.5))
    session = AudiometrySession()
    joint = estimator.initialize_joint_state(session)

    assert len(joint.frequencies_hz) == 8
    for freq in joint.frequencies_hz:
        posterior = np.array(joint.posterior_probs_by_freq[freq], dtype=np.float64)
        assert posterior.shape[0] == len(joint.threshold_grid_db_hl)
        assert np.isfinite(posterior).all()
        assert abs(float(np.sum(posterior)) - 1.0) < 1e-9


def test_joint_update_changes_target_frequency_and_keeps_normalized() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.15, guess_rate=0.5))
    session = AudiometrySession()
    joint = estimator.initialize_joint_state(session)
    freq = joint.frequencies_hz[0]
    prior_mean = joint.posterior_mean_by_freq[freq]

    updated = estimator.update_joint_state(joint, frequency_hz=freq, amplitude_db_hl=45.0, heard=True)
    posterior = np.array(updated.posterior_probs_by_freq[freq], dtype=np.float64)

    assert abs(float(np.sum(posterior)) - 1.0) < 1e-9
    assert np.isfinite(posterior).all()
    assert updated.posterior_mean_by_freq[freq] != prior_mean


def test_joint_frequency_and_amplitude_selection_valid() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig())
    session = AudiometrySession()
    joint = estimator.initialize_joint_state(session)

    freq, amp, ig = estimator.select_next_stimulus(joint)
    assert freq in joint.frequencies_hz
    assert amp in joint.candidate_amplitudes_db_hl
    assert math.isfinite(ig)


def test_validation_loop_joint_path_end_to_end_outputs_finite() -> None:
    result = run_single_validation(
        "sloping_high_frequency",
        engine_cfg=AudiometryEngineConfig(estimator_mode="bayesian", lapse_rate=0.12, guess_rate=0.5),
        slope=0.35,
        seed=5,
        jitter_std=0.0,
    )
    assert math.isfinite(result.mean_abs_error)
    assert result.total_trials > 0
    assert len(result.uncertainty_db) == 8
    assert len(result.trial_count_by_freq) == 8


def test_profile_uncertainty_and_reliability_serialization(tmp_path: Path) -> None:
    _, profile = run_hearing_test(
        AudiometryEngineConfig(estimator_mode="bayesian", lapse_rate=0.1, guess_rate=0.5, max_trials_per_frequency=10, min_trials_per_frequency=4),
        mode="simulated",
        ground_truth_audiogram=[15, 20, 25, 30, 40, 50, 60, 70],
        seed=9,
        verbose=False,
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)

    assert len(loaded.uncertainty) == 8
    assert len(loaded.posterior_variance) == 8
    assert loaded.reliability_score is not None
    assert loaded.lapse_rate_assumed is not None
