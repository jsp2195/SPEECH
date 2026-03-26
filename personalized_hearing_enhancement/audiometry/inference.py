from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from personalized_hearing_enhancement.audiometry.session import AudiometrySession, FrequencyState

_EPS = 1e-12


@dataclass
class StaircaseConfig:
    start_amplitude_db_hl: float = 40.0
    step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4
    min_db_hl: float = 0.0
    max_db_hl: float = 100.0


@dataclass
class BayesianConfig:
    threshold_min_db_hl: float = 0.0
    threshold_max_db_hl: float = 100.0
    threshold_step_db: float = 2.0
    psychometric_slope: float = 0.35
    lapse_rate: float = 0.0
    guess_rate: float = 0.5
    candidate_amplitudes_db_hl: list[float] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    max_trials_per_frequency: int = 18
    min_trials_per_frequency: int = 6
    variance_stop_threshold: float = 9.0
    entropy_stop_threshold: float = 1.4
    low_reliability_threshold: float = 0.45


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


class BayesianThresholdEstimator:
    def __init__(self, cfg: BayesianConfig):
        self.cfg = cfg
        if cfg.psychometric_slope <= 0:
            raise ValueError("psychometric_slope must be > 0")
        if cfg.threshold_step_db <= 0:
            raise ValueError("threshold_step_db must be > 0")
        if not (0.0 <= cfg.lapse_rate < 1.0):
            raise ValueError("lapse_rate must be in [0, 1)")
        if not (0.0 <= cfg.guess_rate <= 1.0):
            raise ValueError("guess_rate must be in [0, 1]")

        self.threshold_grid = np.arange(
            cfg.threshold_min_db_hl,
            cfg.threshold_max_db_hl + cfg.threshold_step_db / 2.0,
            cfg.threshold_step_db,
            dtype=np.float64,
        )
        self.candidate_amplitudes = np.array(sorted({float(x) for x in cfg.candidate_amplitudes_db_hl}), dtype=np.float64)
        if self.candidate_amplitudes.size == 0:
            raise ValueError("candidate_amplitudes_db_hl must have at least one level")

    def initialize_frequency_state(self, state: FrequencyState) -> None:
        posterior = np.full_like(self.threshold_grid, fill_value=1.0 / self.threshold_grid.size, dtype=np.float64)
        self._write_state_summary(state, posterior)
        best_amp, _ = self.select_next_amplitude(state)
        state.current_db_hl = float(best_amp)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        pos_mask = x >= 0
        out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[~pos_mask])
        out[~pos_mask] = exp_x / (1.0 + exp_x)
        return out

    def hearing_probability_given_theta(self, amplitude_db_hl: float, theta_grid: np.ndarray | None = None) -> np.ndarray:
        theta = self.threshold_grid if theta_grid is None else theta_grid
        clean = self._sigmoid(self.cfg.psychometric_slope * (float(amplitude_db_hl) - theta))
        p_heard = (1.0 - self.cfg.lapse_rate) * clean + self.cfg.lapse_rate * self.cfg.guess_rate
        return np.clip(p_heard, _EPS, 1.0 - _EPS)

    def _posterior_from_response(self, posterior: np.ndarray, amplitude_db_hl: float, heard: bool) -> np.ndarray:
        p_heard = self.hearing_probability_given_theta(amplitude_db_hl)
        likelihood = p_heard if heard else (1.0 - p_heard)
        weighted = posterior * np.clip(likelihood, _EPS, 1.0)
        normalizer = float(np.sum(weighted))
        if normalizer <= _EPS:
            return posterior.copy()
        return weighted / normalizer

    def _entropy(self, posterior: np.ndarray) -> float:
        p = np.clip(posterior, _EPS, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _credible_interval(self, posterior: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
        cdf = np.cumsum(posterior)
        alpha = (1.0 - mass) / 2.0
        lo_idx = int(np.searchsorted(cdf, alpha, side="left"))
        hi_idx = int(np.searchsorted(cdf, 1.0 - alpha, side="left"))
        hi_idx = min(hi_idx, len(self.threshold_grid) - 1)
        return float(self.threshold_grid[lo_idx]), float(self.threshold_grid[hi_idx])

    def _stats(self, posterior: np.ndarray) -> dict[str, float | list[float]]:
        mean = float(np.sum(self.threshold_grid * posterior))
        var = float(np.sum(((self.threshold_grid - mean) ** 2) * posterior))
        entropy = self._entropy(posterior)
        map_idx = int(np.argmax(posterior))
        ci_lo, ci_hi = self._credible_interval(posterior, mass=0.95)
        return {
            "mean": mean,
            "variance": var,
            "entropy": entropy,
            "map": float(self.threshold_grid[map_idx]),
            "ci95": [ci_lo, ci_hi],
        }

    def _read_posterior(self, state: FrequencyState) -> np.ndarray:
        if not state.posterior_probs:
            posterior = np.full_like(self.threshold_grid, fill_value=1.0 / self.threshold_grid.size, dtype=np.float64)
        else:
            posterior = np.array(state.posterior_probs, dtype=np.float64)
            posterior = posterior / max(float(np.sum(posterior)), _EPS)
        return posterior

    def _write_state_summary(self, state: FrequencyState, posterior: np.ndarray) -> None:
        stats = self._stats(posterior)
        state.posterior_grid_db_hl = [float(x) for x in self.threshold_grid.tolist()]
        state.posterior_probs = [float(x) for x in posterior.tolist()]
        state.posterior_mean_db_hl = float(stats["mean"])
        state.posterior_variance_db2 = float(stats["variance"])
        state.posterior_entropy = float(stats["entropy"])
        state.posterior_map_db_hl = float(stats["map"])
        state.posterior_ci95 = [float(stats["ci95"][0]), float(stats["ci95"][1])]  # type: ignore[index]
        state.threshold_estimate_db_hl = float(stats["mean"])
        state.uncertainty_db = float(math.sqrt(max(stats["variance"], 0.0)))

    def expected_information_gain(self, posterior: np.ndarray, amplitude_db_hl: float) -> tuple[float, float]:
        current_entropy = self._entropy(posterior)
        p_heard_by_theta = self.hearing_probability_given_theta(amplitude_db_hl)
        p_heard = float(np.sum(posterior * p_heard_by_theta))
        post_heard = self._posterior_from_response(posterior, amplitude_db_hl, heard=True)
        post_not = self._posterior_from_response(posterior, amplitude_db_hl, heard=False)
        expected_entropy = p_heard * self._entropy(post_heard) + (1.0 - p_heard) * self._entropy(post_not)
        ig = current_entropy - expected_entropy
        return float(ig), float(current_entropy)

    def select_next_amplitude(self, state: FrequencyState) -> tuple[float, float]:
        posterior = self._read_posterior(state)
        best_amp = float(self.candidate_amplitudes[0])
        best_ig = -float("inf")
        for amp in self.candidate_amplitudes:
            ig, _ = self.expected_information_gain(posterior, float(amp))
            if ig > best_ig:
                best_ig = ig
                best_amp = float(amp)
        return best_amp, float(best_ig)

    def should_stop(self, state: FrequencyState) -> bool:
        if len(state.trials) >= self.cfg.max_trials_per_frequency:
            state.max_trials_reached = True
            return True
        if len(state.trials) < self.cfg.min_trials_per_frequency:
            return False
        variance = float(state.posterior_variance_db2 if state.posterior_variance_db2 is not None else np.inf)
        entropy = float(state.posterior_entropy if state.posterior_entropy is not None else np.inf)
        return variance <= self.cfg.variance_stop_threshold or entropy <= self.cfg.entropy_stop_threshold

    def record_response(self, session: AudiometrySession, frequency_hz: int, heard: bool) -> FrequencyState:
        state = session.state_for(frequency_hz)
        if not state.posterior_probs:
            self.initialize_frequency_state(state)

        trial_idx = len(state.trials) + 1
        state.trials.append(
            {
                "trial": float(trial_idx),
                "amplitude_db_hl": float(state.current_db_hl),
                "heard": bool(heard),
            }
        )

        prior = self._read_posterior(state)
        posterior = self._posterior_from_response(prior, amplitude_db_hl=float(state.current_db_hl), heard=bool(heard))
        self._write_state_summary(state, posterior)

        if self.should_stop(state):
            state.complete = True
            return state

        next_amp, _ = self.select_next_amplitude(state)
        state.current_db_hl = float(next_amp)
        return state

    def summarize(self, session: AudiometrySession) -> dict[int, dict[str, float | int | list[float]]]:
        summary: dict[int, dict[str, float | int | list[float]]] = {}
        for f in session.frequencies_hz:
            s = session.state_for(f)
            summary[f] = {
                "estimated_threshold_db_hl": float(s.posterior_mean_db_hl if s.posterior_mean_db_hl is not None else s.current_db_hl),
                "posterior_variance_db2": float(s.posterior_variance_db2 if s.posterior_variance_db2 is not None else float("nan")),
                "posterior_entropy": float(s.posterior_entropy if s.posterior_entropy is not None else float("nan")),
                "trial_count": len(s.trials),
                "uncertainty_db": float(s.uncertainty_db if s.uncertainty_db is not None else float("nan")),
                "ci95": s.posterior_ci95 or [float("nan"), float("nan")],
            }
        return summary
