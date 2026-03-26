from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

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


@dataclass
class JointAudiometryState:
    frequencies_hz: list[int]
    threshold_grid_db_hl: list[float]
    candidate_amplitudes_db_hl: list[float]
    posterior_probs_by_freq: dict[int, list[float]]
    posterior_mean_by_freq: dict[int, float]
    posterior_variance_by_freq: dict[int, float]
    posterior_entropy_by_freq: dict[int, float]
    posterior_ci95_by_freq: dict[int, list[float]]
    trial_count_by_freq: dict[int, int]
    completed_by_freq: dict[int, bool]
    response_history: list[dict[str, float | bool | int]] = field(default_factory=list)
    shared_latents: dict[str, float | None] = field(default_factory=lambda: {"device_gain_db": None})
    shared_metadata: dict[str, float | str | bool] = field(default_factory=dict)


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
        ci_lo, ci_hi = self._credible_interval(posterior, mass=0.95)
        return {
            "mean": mean,
            "variance": var,
            "entropy": entropy,
            "ci95": [ci_lo, ci_hi],
        }

    def initialize_joint_state(self, session: AudiometrySession) -> JointAudiometryState:
        uniform = np.full_like(self.threshold_grid, fill_value=1.0 / self.threshold_grid.size, dtype=np.float64)
        state = JointAudiometryState(
            frequencies_hz=list(session.frequencies_hz),
            threshold_grid_db_hl=[float(v) for v in self.threshold_grid.tolist()],
            candidate_amplitudes_db_hl=[float(v) for v in self.candidate_amplitudes.tolist()],
            posterior_probs_by_freq={f: [float(v) for v in uniform.tolist()] for f in session.frequencies_hz},
            posterior_mean_by_freq={f: float(np.mean(self.threshold_grid)) for f in session.frequencies_hz},
            posterior_variance_by_freq={f: float(np.var(self.threshold_grid)) for f in session.frequencies_hz},
            posterior_entropy_by_freq={f: float(self._entropy(uniform)) for f in session.frequencies_hz},
            posterior_ci95_by_freq={f: [float(self.threshold_grid[0]), float(self.threshold_grid[-1])] for f in session.frequencies_hz},
            trial_count_by_freq={f: 0 for f in session.frequencies_hz},
            completed_by_freq={f: False for f in session.frequencies_hz},
            shared_metadata={"future_latent_config": "device_gain_placeholder"},
        )
        session.joint_inference_state = asdict(state)
        self._sync_session_frequency_states(session, state)
        return state

    def _sync_session_frequency_states(self, session: AudiometrySession, joint_state: JointAudiometryState) -> None:
        for freq in session.frequencies_hz:
            fs = session.state_for(freq)
            fs.posterior_grid_db_hl = list(joint_state.threshold_grid_db_hl)
            fs.posterior_probs = list(joint_state.posterior_probs_by_freq[freq])
            fs.posterior_mean_db_hl = float(joint_state.posterior_mean_by_freq[freq])
            fs.posterior_variance_db2 = float(joint_state.posterior_variance_by_freq[freq])
            fs.posterior_entropy = float(joint_state.posterior_entropy_by_freq[freq])
            fs.posterior_ci95 = list(joint_state.posterior_ci95_by_freq[freq])
            fs.threshold_estimate_db_hl = float(joint_state.posterior_mean_by_freq[freq])
            fs.uncertainty_db = float(math.sqrt(max(joint_state.posterior_variance_by_freq[freq], 0.0)))
            fs.complete = bool(joint_state.completed_by_freq[freq])

    def expected_information_gain(self, posterior: np.ndarray, amplitude_db_hl: float) -> tuple[float, float]:
        current_entropy = self._entropy(posterior)
        p_heard_by_theta = self.hearing_probability_given_theta(amplitude_db_hl)
        p_heard = float(np.sum(posterior * p_heard_by_theta))
        post_heard = self._posterior_from_response(posterior, amplitude_db_hl, heard=True)
        post_not = self._posterior_from_response(posterior, amplitude_db_hl, heard=False)
        expected_entropy = p_heard * self._entropy(post_heard) + (1.0 - p_heard) * self._entropy(post_not)
        return float(current_entropy - expected_entropy), float(current_entropy)

    def _should_stop_frequency(self, joint_state: JointAudiometryState, frequency_hz: int) -> bool:
        trials = joint_state.trial_count_by_freq[frequency_hz]
        if trials >= self.cfg.max_trials_per_frequency:
            return True
        if trials < self.cfg.min_trials_per_frequency:
            return False
        var = joint_state.posterior_variance_by_freq[frequency_hz]
        entropy = joint_state.posterior_entropy_by_freq[frequency_hz]
        return var <= self.cfg.variance_stop_threshold or entropy <= self.cfg.entropy_stop_threshold

    def _select_amplitude_for_frequency(self, joint_state: JointAudiometryState, frequency_hz: int) -> tuple[float, float]:
        posterior = np.array(joint_state.posterior_probs_by_freq[frequency_hz], dtype=np.float64)
        best_amp = float(self.candidate_amplitudes[0])
        best_ig = -float("inf")
        for amp in self.candidate_amplitudes:
            ig, _ = self.expected_information_gain(posterior, float(amp))
            if ig > best_ig:
                best_ig = ig
                best_amp = float(amp)
        return best_amp, float(best_ig)

    def select_next_stimulus(self, joint_state: JointAudiometryState) -> tuple[int, float, float]:
        unfinished = [f for f in joint_state.frequencies_hz if not joint_state.completed_by_freq[f]]
        if not unfinished:
            raise RuntimeError("Joint state is complete")
        selected_freq = max(unfinished, key=lambda f: (joint_state.posterior_entropy_by_freq[f], joint_state.posterior_variance_by_freq[f]))
        amp, ig = self._select_amplitude_for_frequency(joint_state, selected_freq)
        return int(selected_freq), float(amp), float(ig)

    def update_joint_state(
        self,
        joint_state: JointAudiometryState,
        *,
        frequency_hz: int,
        amplitude_db_hl: float,
        heard: bool,
    ) -> JointAudiometryState:
        freq = int(frequency_hz)
        posterior = np.array(joint_state.posterior_probs_by_freq[freq], dtype=np.float64)
        posterior = self._posterior_from_response(posterior, amplitude_db_hl=float(amplitude_db_hl), heard=bool(heard))
        stats = self._stats(posterior)

        joint_state.posterior_probs_by_freq[freq] = [float(v) for v in posterior.tolist()]
        joint_state.posterior_mean_by_freq[freq] = float(stats["mean"])
        joint_state.posterior_variance_by_freq[freq] = float(stats["variance"])
        joint_state.posterior_entropy_by_freq[freq] = float(stats["entropy"])
        joint_state.posterior_ci95_by_freq[freq] = [float(stats["ci95"][0]), float(stats["ci95"][1])]  # type: ignore[index]
        joint_state.trial_count_by_freq[freq] += 1
        joint_state.response_history.append(
            {
                "frequency_hz": int(freq),
                "amplitude_db_hl": float(amplitude_db_hl),
                "heard": bool(heard),
                "trial_at_frequency": int(joint_state.trial_count_by_freq[freq]),
            }
        )
        joint_state.completed_by_freq[freq] = self._should_stop_frequency(joint_state, freq)
        return joint_state

    def summarize_joint_state(self, joint_state: JointAudiometryState) -> dict[int, dict[str, float | int | list[float]]]:
        out: dict[int, dict[str, float | int | list[float]]] = {}
        for f in joint_state.frequencies_hz:
            out[f] = {
                "estimated_threshold_db_hl": float(joint_state.posterior_mean_by_freq[f]),
                "posterior_variance_db2": float(joint_state.posterior_variance_by_freq[f]),
                "posterior_entropy": float(joint_state.posterior_entropy_by_freq[f]),
                "trial_count": int(joint_state.trial_count_by_freq[f]),
                "uncertainty_db": float(math.sqrt(max(joint_state.posterior_variance_by_freq[f], 0.0))),
                "ci95": list(joint_state.posterior_ci95_by_freq[f]),
            }
        return out

    def get_threshold_estimates(self, joint_state: JointAudiometryState) -> list[float]:
        return [float(joint_state.posterior_mean_by_freq[f]) for f in joint_state.frequencies_hz]

    def get_uncertainty_estimates(self, joint_state: JointAudiometryState) -> list[float]:
        return [float(math.sqrt(max(joint_state.posterior_variance_by_freq[f], 0.0))) for f in joint_state.frequencies_hz]

    def is_complete(self, joint_state: JointAudiometryState) -> bool:
        return all(bool(v) for v in joint_state.completed_by_freq.values())
