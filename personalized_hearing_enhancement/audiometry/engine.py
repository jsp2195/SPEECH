from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from personalized_hearing_enhancement.audiometry.inference import (
    BayesianConfig,
    BayesianThresholdEstimator,
    StaircaseConfig,
    StaircaseEstimator,
)
from personalized_hearing_enhancement.audiometry.profiles import HearingProfile
from personalized_hearing_enhancement.audiometry.session import AudiometrySession


@dataclass
class AudiometryEngineConfig:
    sample_rate: int = 16000
    estimator_mode: str = "bayesian"
    # Legacy staircase settings
    start_amplitude_db_hl: float = 40.0
    step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4
    # Bayesian settings
    threshold_min_db_hl: float = 0.0
    threshold_max_db_hl: float = 100.0
    threshold_step_db: float = 2.0
    psychometric_slope: float = 0.35
    candidate_amplitudes_db_hl: list[float] = field(
        default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    variance_stop_threshold: float = 9.0
    entropy_stop_threshold: float = 1.4
    min_trials_per_frequency: int = 6


@dataclass
class SimulatedResponderConfig:
    psychometric_slope: float = 0.35


def logistic_hear_probability(amplitude_db_hl: float, threshold_db_hl: float, slope: float = 0.35) -> float:
    if slope <= 0:
        raise ValueError("slope must be positive")
    x = slope * (amplitude_db_hl - threshold_db_hl)
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _simulated_heard(amplitude_db_hl: float, threshold_db_hl: float, rng: random.Random, slope: float) -> tuple[bool, float]:
    prob = logistic_hear_probability(amplitude_db_hl, threshold_db_hl, slope=slope)
    return (rng.random() < prob), prob


def _to_heard(response: str) -> bool:
    token = response.strip().lower()
    if token in {"y", "yes", "heard", "1"}:
        return True
    if token in {"n", "no", "not", "not_heard", "0"}:
        return False
    raise ValueError("Response must be y/n or heard/not")


def run_hearing_test(
    cfg: AudiometryEngineConfig,
    *,
    mode: str = "interactive",
    ground_truth_audiogram: list[float] | None = None,
    seed: int | None = None,
    save_progress_path: str | None = None,
    simulated_responder: SimulatedResponderConfig | None = None,
    response_callback: Callable[[int, float, int], bool] | None = None,
    verbose: bool = True,
) -> tuple[AudiometrySession, HearingProfile]:
    if mode not in {"interactive", "simulated"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "simulated" and ground_truth_audiogram is None:
        raise ValueError("ground_truth_audiogram is required in simulated mode")

    rng = random.Random(seed)
    responder_cfg = simulated_responder or SimulatedResponderConfig()

    session = AudiometrySession(
        start_amplitude_db_hl=cfg.start_amplitude_db_hl,
        initial_step_size_db=cfg.step_size_db,
        min_step_size_db=cfg.min_step_size_db,
        max_trials_per_frequency=cfg.max_trials_per_frequency,
        max_reversals=cfg.max_reversals,
        seed=seed,
        source="simulated" if mode == "simulated" else "interactive",
    )

    estimator_mode = cfg.estimator_mode.lower()
    if estimator_mode == "bayesian":
        estimator: BayesianThresholdEstimator | StaircaseEstimator = BayesianThresholdEstimator(
            BayesianConfig(
                threshold_min_db_hl=cfg.threshold_min_db_hl,
                threshold_max_db_hl=cfg.threshold_max_db_hl,
                threshold_step_db=cfg.threshold_step_db,
                psychometric_slope=cfg.psychometric_slope,
                candidate_amplitudes_db_hl=cfg.candidate_amplitudes_db_hl,
                max_trials_per_frequency=cfg.max_trials_per_frequency,
                min_trials_per_frequency=cfg.min_trials_per_frequency,
                variance_stop_threshold=cfg.variance_stop_threshold,
                entropy_stop_threshold=cfg.entropy_stop_threshold,
            )
        )
    elif estimator_mode == "staircase":
        estimator = StaircaseEstimator(
            StaircaseConfig(
                start_amplitude_db_hl=cfg.start_amplitude_db_hl,
                step_size_db=cfg.step_size_db,
                min_step_size_db=cfg.min_step_size_db,
                max_trials_per_frequency=cfg.max_trials_per_frequency,
                max_reversals=cfg.max_reversals,
            )
        )
    else:
        raise ValueError(f"Unsupported estimator_mode: {cfg.estimator_mode}")

    for idx, freq in enumerate(session.frequencies_hz):
        session.active_frequency_index = idx
        state = session.state_for(freq)
        if estimator_mode == "bayesian":
            assert isinstance(estimator, BayesianThresholdEstimator)
            estimator.initialize_frequency_state(state)

        if verbose:
            print(f"\nTesting {freq} Hz")

        while not state.complete:
            if verbose:
                if estimator_mode == "bayesian":
                    print(
                        f"  Trial {len(state.trials)+1}: level={state.current_db_hl:.1f} dB HL | "
                        f"mean={state.posterior_mean_db_hl:.1f} dB HL | "
                        f"std={state.uncertainty_db:.2f} dB | entropy={state.posterior_entropy:.3f}"
                    )
                else:
                    print(f"  Trial {len(state.trials)+1}: level={state.current_db_hl:.1f} dB HL")

            if mode == "simulated":
                if response_callback is not None:
                    heard = bool(response_callback(freq, float(state.current_db_hl), len(state.trials)))
                    heard_prob = None
                else:
                    gt = float(ground_truth_audiogram[idx])
                    heard, heard_prob = _simulated_heard(
                        state.current_db_hl,
                        gt,
                        rng,
                        slope=responder_cfg.psychometric_slope,
                    )
                if verbose:
                    if heard_prob is None:
                        print(f"    simulated response: {'heard' if heard else 'not heard'}")
                    else:
                        print(
                            f"    simulated response: {'heard' if heard else 'not heard'} "
                            f"(p_heard={heard_prob:.3f})"
                        )
            else:
                heard = _to_heard(input("    Heard tone? [y/n]: "))

            state = estimator.record_response(session, freq, heard)
            if save_progress_path:
                Path(save_progress_path).write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

        if verbose:
            if estimator_mode == "bayesian":
                print(
                    f"  Completed {freq} Hz -> threshold={state.threshold_estimate_db_hl:.1f} dB HL "
                    f"(std={state.uncertainty_db:.2f} dB, entropy={state.posterior_entropy:.3f}, trials={len(state.trials)})"
                )
            else:
                print(
                    f"  Completed {freq} Hz -> threshold={state.threshold_estimate_db_hl:.1f} dB HL "
                    f"(uncertainty ±{state.uncertainty_db:.1f}, trials={len(state.trials)})"
                )

    summary = estimator.summarize(session)
    thresholds = [float(summary[f]["estimated_threshold_db_hl"]) for f in session.frequencies_hz]
    uncertainty = [float(summary[f]["uncertainty_db"]) for f in session.frequencies_hz]

    metadata: dict[str, str | int | float | bool] = {
        "max_trials_per_frequency": cfg.max_trials_per_frequency,
        "psychometric_slope": responder_cfg.psychometric_slope,
        "audiometry_mode": estimator_mode,
    }
    if estimator_mode == "staircase":
        metadata["max_reversals"] = cfg.max_reversals

    profile = HearingProfile(
        frequencies=list(session.frequencies_hz),
        thresholds_db=thresholds,
        uncertainty=uncertainty,
        source="simulated" if mode == "simulated" else "estimated",
        sample_rate=cfg.sample_rate,
        metadata=metadata,
    )

    if estimator_mode == "bayesian":
        profile.posterior_variance = [float(summary[f]["posterior_variance_db2"]) for f in session.frequencies_hz]
        profile.posterior_entropy = [float(summary[f]["posterior_entropy"]) for f in session.frequencies_hz]
        profile.credible_interval_width = [
            float(summary[f]["ci95"][1] - summary[f]["ci95"][0])  # type: ignore[index]
            for f in session.frequencies_hz
        ]

    if verbose:
        print("\nEstimated thresholds (dB HL):", thresholds)
    return session, profile
