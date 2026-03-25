from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from personalized_hearing_enhancement.audiometry.inference import StaircaseConfig, StaircaseEstimator
from personalized_hearing_enhancement.audiometry.profiles import HearingProfile
from personalized_hearing_enhancement.audiometry.session import AudiometrySession


@dataclass
class AudiometryEngineConfig:
    sample_rate: int = 16000
    start_amplitude_db_hl: float = 40.0
    step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4


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
) -> tuple[AudiometrySession, HearingProfile]:
    if mode not in {"interactive", "simulated"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "simulated" and ground_truth_audiogram is None:
        raise ValueError("ground_truth_audiogram is required in simulated mode")

    rng = random.Random(seed)
    session = AudiometrySession(
        start_amplitude_db_hl=cfg.start_amplitude_db_hl,
        initial_step_size_db=cfg.step_size_db,
        min_step_size_db=cfg.min_step_size_db,
        max_trials_per_frequency=cfg.max_trials_per_frequency,
        max_reversals=cfg.max_reversals,
        seed=seed,
        source="simulated" if mode == "simulated" else "interactive",
    )
    estimator = StaircaseEstimator(
        StaircaseConfig(
            start_amplitude_db_hl=cfg.start_amplitude_db_hl,
            step_size_db=cfg.step_size_db,
            min_step_size_db=cfg.min_step_size_db,
            max_trials_per_frequency=cfg.max_trials_per_frequency,
            max_reversals=cfg.max_reversals,
        )
    )

    for idx, freq in enumerate(session.frequencies_hz):
        session.active_frequency_index = idx
        state = session.state_for(freq)
        print(f"\nTesting {freq} Hz")
        while not state.complete:
            print(f"  Trial {len(state.trials)+1}: level={state.current_db_hl:.1f} dB HL")
            if mode == "simulated":
                gt = float(ground_truth_audiogram[idx])
                heard_prob = 1.0 if state.current_db_hl >= gt else 0.0
                heard = rng.random() < heard_prob
                print(f"    simulated response: {'heard' if heard else 'not heard'}")
            else:
                heard = _to_heard(input("    Heard tone? [y/n]: "))

            state = estimator.record_response(session, freq, heard)
            if save_progress_path:
                Path(save_progress_path).write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

        print(
            f"  Completed {freq} Hz -> threshold={state.threshold_estimate_db_hl:.1f} dB HL "
            f"(uncertainty ±{state.uncertainty_db:.1f}, trials={len(state.trials)})"
        )

    summary = estimator.summarize(session)
    thresholds = [float(summary[f]["estimated_threshold_db_hl"]) for f in session.frequencies_hz]
    uncertainty = [float(summary[f]["uncertainty_db"]) for f in session.frequencies_hz]
    profile = HearingProfile(
        frequencies=list(session.frequencies_hz),
        thresholds_db=thresholds,
        uncertainty=uncertainty,
        source="simulated" if mode == "simulated" else "estimated",
        sample_rate=cfg.sample_rate,
        metadata={"max_trials_per_frequency": cfg.max_trials_per_frequency, "max_reversals": cfg.max_reversals},
    )
    print("\nEstimated thresholds (dB HL):", thresholds)
    return session, profile
