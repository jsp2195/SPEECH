from __future__ import annotations

from dataclasses import dataclass

from personalized_hearing_enhancement.audiometry.session import AudiometrySession, FrequencyState


@dataclass
class StaircaseConfig:
    start_amplitude_db_hl: float = 40.0
    step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4
    min_db_hl: float = 0.0
    max_db_hl: float = 100.0


class StaircaseEstimator:
    def __init__(self, cfg: StaircaseConfig):
        self.cfg = cfg

    def record_response(self, session: AudiometrySession, frequency_hz: int, heard: bool) -> FrequencyState:
        state = session.state_for(frequency_hz)
        prev_heard = None if not state.trials else bool(state.trials[-1]["heard"])
        state.trials.append(
            {
                "trial": float(len(state.trials) + 1),
                "amplitude_db_hl": float(state.current_db_hl),
                "heard": bool(heard),
            }
        )

        if prev_heard is not None and bool(prev_heard) != bool(heard):
            state.reversals += 1
            state.step_size_db = max(self.cfg.min_step_size_db, state.step_size_db / 2.0)

        delta = -state.step_size_db if heard else state.step_size_db
        next_level = state.current_db_hl + delta
        state.current_db_hl = float(min(self.cfg.max_db_hl, max(self.cfg.min_db_hl, next_level)))

        if len(state.trials) >= self.cfg.max_trials_per_frequency:
            state.max_trials_reached = True
        if state.reversals >= self.cfg.max_reversals or state.max_trials_reached:
            state.complete = True
            state.threshold_estimate_db_hl = self._estimate_threshold(state)
            state.uncertainty_db = float(state.step_size_db)

        return state

    @staticmethod
    def _estimate_threshold(state: FrequencyState) -> float:
        heard_levels = [float(t["amplitude_db_hl"]) for t in state.trials if bool(t["heard"])]
        not_heard_levels = [float(t["amplitude_db_hl"]) for t in state.trials if not bool(t["heard"])]
        if heard_levels and not_heard_levels:
            return float((max(heard_levels) + min(not_heard_levels)) / 2.0)
        if heard_levels:
            return float(sum(heard_levels[-3:]) / min(3, len(heard_levels)))
        if not_heard_levels:
            return float(sum(not_heard_levels[-3:]) / min(3, len(not_heard_levels)))
        return float(state.current_db_hl)

    def summarize(self, session: AudiometrySession) -> dict[int, dict[str, float | int]]:
        summary: dict[int, dict[str, float | int]] = {}
        for f in session.frequencies_hz:
            s = session.state_for(f)
            est = s.threshold_estimate_db_hl
            if est is None:
                est = self._estimate_threshold(s)
            summary[f] = {
                "estimated_threshold_db_hl": float(est),
                "trial_count": len(s.trials),
                "reversals": s.reversals,
                "uncertainty_db": float(s.step_size_db),
            }
        return summary
