from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.inference import BayesianConfig, BayesianThresholdEstimator
from personalized_hearing_enhancement.audiometry.profiles import load_profile, save_profile
from personalized_hearing_enhancement.audiometry.session import AudiometrySession
from personalized_hearing_enhancement.audiometry.validation import run_single_validation


def test_device_gain_posterior_initialization_valid() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(infer_device_gain=True))
    joint = estimator.initialize_joint_state(AudiometrySession())
    gain_post = np.array(joint.device_gain_posterior, dtype=np.float64)
    assert gain_post.shape[0] == len(joint.device_gain_grid_db)
    assert np.isfinite(gain_post).all()
    assert abs(float(np.sum(gain_post)) - 1.0) < 1e-9


def test_gain_aware_likelihood_reduces_to_gain_zero_case() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.0, guess_rate=0.5, infer_device_gain=True))
    theta = np.array([40.0])
    p0 = estimator.hearing_probability_given_theta_gain(40.0, theta[:, None], np.array([0.0])[None, :])[0, 0]
    pplus = estimator.hearing_probability_given_theta_gain(40.0, theta[:, None], np.array([6.0])[None, :])[0, 0]
    assert 0.0 < p0 < 1.0
    assert pplus > p0


def test_joint_update_keeps_threshold_and_gain_normalized() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(infer_device_gain=True))
    joint = estimator.initialize_joint_state(AudiometrySession())
    freq = joint.frequencies_hz[0]
    joint = estimator.update_joint_state(joint, frequency_hz=freq, amplitude_db_hl=45.0, heard=True)
    assert abs(float(np.sum(np.array(joint.posterior_probs_by_freq[freq]))) - 1.0) < 1e-9
    assert abs(float(np.sum(np.array(joint.device_gain_posterior))) - 1.0) < 1e-9


def test_gain_aware_stimulus_selection_valid() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(infer_device_gain=True))
    joint = estimator.initialize_joint_state(AudiometrySession())
    freq, amp, ig = estimator.select_next_stimulus(joint)
    assert freq in joint.frequencies_hz
    assert amp in joint.candidate_amplitudes_db_hl
    assert math.isfinite(ig)


def test_validation_with_unknown_gain_runs_end_to_end() -> None:
    result = run_single_validation(
        "normal",
        engine_cfg=AudiometryEngineConfig(estimator_mode="bayesian", infer_device_gain=True),
        true_device_gain_db=6.0,
        slope=0.35,
        seed=5,
        jitter_std=0.0,
    )
    assert math.isfinite(result.mean_abs_error)
    assert math.isfinite(result.device_gain_abs_error)
    assert result.total_trials > 0


def test_profile_gain_metadata_serialization(tmp_path: Path) -> None:
    _, profile = run_hearing_test(
        AudiometryEngineConfig(estimator_mode="bayesian", infer_device_gain=True, max_trials_per_frequency=10, min_trials_per_frequency=4),
        mode="simulated",
        ground_truth_audiogram=[15, 20, 25, 30, 40, 50, 60, 70],
        seed=9,
        verbose=False,
    )
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.estimated_device_gain_db is not None
    assert loaded.device_gain_uncertainty is not None
