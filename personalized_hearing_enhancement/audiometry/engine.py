from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from personalized_hearing_enhancement.audiometry.inference import BayesianConfig, BayesianThresholdEstimator, StaircaseConfig, StaircaseEstimator
from personalized_hearing_enhancement.audiometry.profiles import HearingProfile, profile_from_joint_summary
from personalized_hearing_enhancement.audiometry.session import AudiometrySession


@dataclass
class AudiometryEngineConfig:
    sample_rate: int = 16000
    estimator_mode: str = "bayesian"
    start_amplitude_db_hl: float = 40.0
    step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4
    threshold_min_db_hl: float = 0.0
    threshold_max_db_hl: float = 100.0
    threshold_step_db: float = 2.0
    psychometric_slope: float = 0.35
    lapse_rate: float = 0.0
    guess_rate: float = 0.5
    infer_device_gain: bool = False
    device_gain_grid_db: list[float] = field(default_factory=lambda: [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18])
    candidate_amplitudes_db_hl: list[float] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    variance_stop_threshold: float = 9.0
    entropy_stop_threshold: float = 1.4
    min_trials_per_frequency: int = 6
    low_reliability_threshold: float = 0.45


@dataclass
class SimulatedResponderConfig:
    psychometric_slope: float = 0.35
    lapse_rate: float = 0.0
    guess_rate: float = 0.5
    true_device_gain_db: float = 0.0
    response_model: str = "clean_logistic"
    inconsistency_rate: float = 0.0
    simulate_fatigue: bool = False
    fatigue_lapse_increment: float = 0.0


def logistic_hear_probability(amplitude_db_hl: float, threshold_db_hl: float, slope: float = 0.35) -> float:
    x = slope * (amplitude_db_hl - threshold_db_hl)
    if x >= 0:
        return float(1.0 / (1.0 + math.exp(-x)))
    ex = math.exp(x)
    return float(ex / (1.0 + ex))


def _simulated_heard(amplitude_db_hl: float, threshold_db_hl: float, rng: random.Random, cfg: SimulatedResponderConfig, trial_idx: int) -> tuple[bool, float]:
    effective = amplitude_db_hl + cfg.true_device_gain_db
    clean = logistic_hear_probability(effective, threshold_db_hl, slope=cfg.psychometric_slope)
    lapse = min(0.95, cfg.lapse_rate + (cfg.fatigue_lapse_increment * max(0, trial_idx - 1) if cfg.simulate_fatigue else 0.0))
    p_heard = clean if cfg.response_model == "clean_logistic" else (1.0 - lapse) * clean + lapse * cfg.guess_rate
    if cfg.inconsistency_rate > 0:
        p_heard = (1.0 - cfg.inconsistency_rate) * p_heard + cfg.inconsistency_rate * (1.0 - p_heard)
    p_heard = float(min(1.0, max(0.0, p_heard)))
    return (rng.random() < p_heard), p_heard


def _to_heard(response: str) -> bool:
    return response.strip().lower() in {"y", "yes", "heard", "1"}


def _update_reliability(session: AudiometrySession, posterior_mean: float, posterior_std: float, amplitude_db_hl: float, heard: bool) -> None:
    session.total_trials += 1
    margin = float(amplitude_db_hl - posterior_mean)
    contradiction = (margin > max(3.0, 0.5 * posterior_std) and not heard) or (margin < -max(3.0, 0.5 * posterior_std) and heard)
    if contradiction:
        session.inconsistency_count += 1
    session.reliability_score = float(max(0.0, 1.0 - 1.5 * (session.inconsistency_count / max(1, session.total_trials))))


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
    if mode == "simulated" and ground_truth_audiogram is None:
        raise ValueError("ground_truth_audiogram is required in simulated mode")

    rng = random.Random(seed)
    responder_cfg = simulated_responder or SimulatedResponderConfig(
        psychometric_slope=cfg.psychometric_slope,
        lapse_rate=cfg.lapse_rate,
        guess_rate=cfg.guess_rate,
        true_device_gain_db=0.0,
        response_model="clean_logistic" if cfg.lapse_rate == 0 else "lapse_logistic",
    )

    session = AudiometrySession(
        start_amplitude_db_hl=cfg.start_amplitude_db_hl,
        initial_step_size_db=cfg.step_size_db,
        min_step_size_db=cfg.min_step_size_db,
        max_trials_per_frequency=cfg.max_trials_per_frequency,
        max_reversals=cfg.max_reversals,
        seed=seed,
        source="simulated" if mode == "simulated" else "interactive",
        lapse_rate_used=cfg.lapse_rate,
        guess_rate_used=cfg.guess_rate,
        fatigue_enabled=responder_cfg.simulate_fatigue,
    )

    if cfg.estimator_mode == "bayesian":
        estimator = BayesianThresholdEstimator(
            BayesianConfig(
                threshold_min_db_hl=cfg.threshold_min_db_hl,
                threshold_max_db_hl=cfg.threshold_max_db_hl,
                threshold_step_db=cfg.threshold_step_db,
                psychometric_slope=cfg.psychometric_slope,
                lapse_rate=cfg.lapse_rate,
                guess_rate=cfg.guess_rate,
                infer_device_gain=cfg.infer_device_gain,
                device_gain_grid_db=cfg.device_gain_grid_db,
                candidate_amplitudes_db_hl=cfg.candidate_amplitudes_db_hl,
                max_trials_per_frequency=cfg.max_trials_per_frequency,
                min_trials_per_frequency=cfg.min_trials_per_frequency,
                variance_stop_threshold=cfg.variance_stop_threshold,
                entropy_stop_threshold=cfg.entropy_stop_threshold,
                low_reliability_threshold=cfg.low_reliability_threshold,
            )
        )
        joint = estimator.initialize_joint_state(session)
        while not estimator.is_complete(joint):
            freq, amp, ig = estimator.select_next_stimulus(joint)
            session.active_frequency_index = session.frequencies_hz.index(freq)
            session.active_amplitude_db_hl = float(amp)
            if verbose:
                print(
                    f"trial={session.total_trials+1} f={freq}Hz a={amp:.1f} mean={joint.posterior_mean_by_freq[freq]:.1f} "
                    f"std={math.sqrt(joint.posterior_variance_by_freq[freq]):.2f} g={joint.device_gain_mean:.2f}±{math.sqrt(joint.device_gain_variance):.2f} ig={ig:.4f}"
                )

            if mode == "simulated":
                if response_callback is not None:
                    heard = bool(response_callback(freq, amp, joint.trial_count_by_freq[freq]))
                else:
                    gt = float(ground_truth_audiogram[session.frequencies_hz.index(freq)])
                    heard, _ = _simulated_heard(amp, gt, rng, responder_cfg, trial_idx=joint.trial_count_by_freq[freq] + 1)
            else:
                heard = _to_heard(input(f"Heard tone at {freq}Hz [y/n]: "))

            _update_reliability(session, joint.posterior_mean_by_freq[freq], math.sqrt(max(joint.posterior_variance_by_freq[freq], 0.0)), amp, heard)
            fs = session.state_for(freq)
            fs.trials.append({"trial": float(len(fs.trials) + 1), "amplitude_db_hl": float(amp), "heard": bool(heard)})
            joint = estimator.update_joint_state(joint, frequency_hz=freq, amplitude_db_hl=amp, heard=heard)
            estimator._sync_session_frequency_states(session, joint)
            session.joint_inference_state = joint.__dict__
            session.device_gain_mean_db = joint.device_gain_mean
            session.device_gain_variance_db2 = joint.device_gain_variance
            session.device_gain_entropy = joint.device_gain_entropy

            if save_progress_path:
                Path(save_progress_path).write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

        summary = estimator.summarize_joint_state(joint)
    else:
        staircase = StaircaseEstimator(StaircaseConfig())
        for idx, freq in enumerate(session.frequencies_hz):
            st = session.state_for(freq)
            while not st.complete:
                heard, _ = _simulated_heard(st.current_db_hl, float(ground_truth_audiogram[idx]), rng, responder_cfg, trial_idx=len(st.trials) + 1)
                st = staircase.record_response(session, freq, heard)
        summary = staircase.summarize(session)

    if session.reliability_score < cfg.low_reliability_threshold:
        session.low_confidence = True

    metadata: dict[str, str | int | float | bool] = {
        "audiometry_mode": cfg.estimator_mode,
        "reliability_score": session.reliability_score,
        "inconsistency_count": session.inconsistency_count,
        "total_trials": session.total_trials,
        "infer_device_gain": cfg.infer_device_gain,
    }

    profile = profile_from_joint_summary(
        list(session.frequencies_hz),
        summary,
        source="simulated" if mode == "simulated" else "estimated",
        sample_rate=cfg.sample_rate,
        metadata=metadata,
        reliability_score=session.reliability_score,
        lapse_rate_assumed=cfg.lapse_rate,
    )
    if cfg.estimator_mode == "bayesian":
        profile.posterior_variance = [float(summary[f]["posterior_variance_db2"]) for f in session.frequencies_hz]
        profile.posterior_entropy = [float(summary[f]["posterior_entropy"]) for f in session.frequencies_hz]
        profile.credible_interval_width = [float(summary[f]["ci95"][1] - summary[f]["ci95"][0]) for f in session.frequencies_hz]  # type: ignore[index]
        profile.estimated_device_gain_db = float(session.device_gain_mean_db or 0.0)
        profile.device_gain_uncertainty = float(math.sqrt(max(session.device_gain_variance_db2 or 0.0, 0.0)))

    return session, profile
