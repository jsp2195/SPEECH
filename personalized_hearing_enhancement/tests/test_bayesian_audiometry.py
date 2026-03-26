from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.inference import BayesianConfig, BayesianThresholdEstimator
from personalized_hearing_enhancement.audiometry.profiles import load_profile, save_profile
from personalized_hearing_enhancement.audiometry.validation import run_single_validation


def test_reliability_aware_likelihood_valid_and_reduces_to_clean() -> None:
    clean_estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.0, guess_rate=0.5))
    noisy_estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.2, guess_rate=0.6))

    amp = 40.0
    p_clean = clean_estimator.hearing_probability_given_theta(amp)
    p_noisy = noisy_estimator.hearing_probability_given_theta(amp)

    assert np.isfinite(p_noisy).all()
    assert np.all((p_noisy >= 0.0) & (p_noisy <= 1.0))
    assert np.allclose(p_clean, clean_estimator._sigmoid(clean_estimator.cfg.psychometric_slope * (amp - clean_estimator.threshold_grid)))


def test_posterior_update_normalized_with_lapse_model() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.15, guess_rate=0.5))
    prior = np.full(estimator.threshold_grid.shape, 1.0 / estimator.threshold_grid.size)
    posterior = estimator._posterior_from_response(prior, 45.0, heard=True)

    assert posterior.shape == prior.shape
    assert np.isfinite(posterior).all()
    assert abs(float(np.sum(posterior)) - 1.0) < 1e-9


def test_expected_information_gain_finite_under_lapse_model() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig(lapse_rate=0.12, guess_rate=0.5))
    prior = np.full(estimator.threshold_grid.shape, 1.0 / estimator.threshold_grid.size)
    ig, entropy = estimator.expected_information_gain(prior, 40.0)
    assert math.isfinite(ig)
    assert math.isfinite(entropy)


def test_validation_loop_noisy_scenario_runs_end_to_end_outputs_finite() -> None:
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
    assert result.reliability_score is not None


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
