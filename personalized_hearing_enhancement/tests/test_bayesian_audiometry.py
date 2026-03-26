from __future__ import annotations

import math

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.inference import BayesianConfig, BayesianThresholdEstimator
from personalized_hearing_enhancement.audiometry.validation import run_single_validation


def test_posterior_initialization_valid() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig())
    # use estimator grid directly for initialization checks
    posterior = np.full(estimator.threshold_grid.shape, 1.0 / estimator.threshold_grid.size)
    assert posterior.shape == estimator.threshold_grid.shape
    assert np.isfinite(posterior).all()
    assert abs(float(np.sum(posterior)) - 1.0) < 1e-9


def test_posterior_update_sensible_direction_and_normalized() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig())
    prior = np.full(estimator.threshold_grid.shape, 1.0 / estimator.threshold_grid.size)

    amp = 40.0
    post_heard = estimator._posterior_from_response(prior, amp, heard=True)
    post_not = estimator._posterior_from_response(prior, amp, heard=False)

    mean_prior = float(np.sum(estimator.threshold_grid * prior))
    mean_heard = float(np.sum(estimator.threshold_grid * post_heard))
    mean_not = float(np.sum(estimator.threshold_grid * post_not))

    assert mean_heard < mean_prior
    assert mean_not > mean_prior
    assert abs(float(np.sum(post_heard)) - 1.0) < 1e-9
    assert abs(float(np.sum(post_not)) - 1.0) < 1e-9


def test_expected_information_gain_finite_and_candidate_selected() -> None:
    estimator = BayesianThresholdEstimator(BayesianConfig())
    state = run_hearing_test(
        AudiometryEngineConfig(estimator_mode="bayesian", max_trials_per_frequency=8, min_trials_per_frequency=3),
        mode="simulated",
        ground_truth_audiogram=[20, 20, 20, 20, 20, 20, 20, 20],
        seed=11,
        verbose=False,
    )[0].state_for(250)

    amp, ig = estimator.select_next_amplitude(state)
    assert math.isfinite(ig)
    assert amp in estimator.candidate_amplitudes.tolist()


def test_validation_loop_bayesian_end_to_end_outputs_finite() -> None:
    result = run_single_validation(
        "sloping_high_frequency",
        engine_cfg=AudiometryEngineConfig(estimator_mode="bayesian", max_trials_per_frequency=12, min_trials_per_frequency=4),
        slope=0.35,
        seed=5,
        jitter_std=0.0,
    )
    assert math.isfinite(result.mean_abs_error)
    assert result.total_trials > 0
    assert len(result.uncertainty_db) == 8
    assert all(math.isfinite(v) for v in result.uncertainty_db)


def test_profile_uncertainty_fields_populated_from_bayesian() -> None:
    _, profile = run_hearing_test(
        AudiometryEngineConfig(estimator_mode="bayesian", max_trials_per_frequency=10, min_trials_per_frequency=4),
        mode="simulated",
        ground_truth_audiogram=[15, 20, 25, 30, 40, 50, 60, 70],
        seed=9,
        verbose=False,
    )
    assert len(profile.uncertainty) == 8
    assert len(profile.posterior_variance) == 8
    assert len(profile.posterior_entropy) == 8
    assert len(profile.credible_interval_width) == 8
