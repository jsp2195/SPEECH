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
    infer_device_gain: bool = False
    device_gain_grid_db: list[float] = field(default_factory=lambda: [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18])
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

    device_gain_grid_db: list[float] = field(default_factory=list)
    device_gain_posterior: list[float] = field(default_factory=list)
    device_gain_mean: float = 0.0
    device_gain_variance: float = 0.0
    device_gain_entropy: float = 0.0

    shared_latents: dict[str, float | None] = field(default_factory=dict)
    shared_metadata: dict[str, float | str | bool] = field(default_factory=dict)


class StaircaseEstimator:
    def __init__(self, cfg: StaircaseConfig):
        self.cfg = cfg

    def record_response(self, session: AudiometrySession, frequency_hz: int, heard: bool) -> FrequencyState:
        state = session.state_for(frequency_hz)
        prev_heard = None if not state.trials else bool(state.trials[-1]["heard"])
        state.trials.append({"trial": float(len(state.trials) + 1), "amplitude_db_hl": float(state.current_db_hl), "heard": bool(heard)})
        if prev_heard is not None and bool(prev_heard) != bool(heard):
            state.reversals += 1
            state.step_size_db = max(self.cfg.min_step_size_db, state.step_size_db / 2.0)
        delta = -state.step_size_db if heard else state.step_size_db
        state.current_db_hl = float(min(self.cfg.max_db_hl, max(self.cfg.min_db_hl, state.current_db_hl + delta)))
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
        out: dict[int, dict[str, float | int]] = {}
        for f in session.frequencies_hz:
            s = session.state_for(f)
            est = s.threshold_estimate_db_hl if s.threshold_estimate_db_hl is not None else self._estimate_threshold(s)
            out[f] = {"estimated_threshold_db_hl": float(est), "trial_count": len(s.trials), "reversals": s.reversals, "uncertainty_db": float(s.step_size_db)}
        return out


class BayesianThresholdEstimator:
    def __init__(self, cfg: BayesianConfig):
        self.cfg = cfg
        self.threshold_grid = np.arange(cfg.threshold_min_db_hl, cfg.threshold_max_db_hl + cfg.threshold_step_db / 2.0, cfg.threshold_step_db, dtype=np.float64)
        self.candidate_amplitudes = np.array(sorted({float(x) for x in cfg.candidate_amplitudes_db_hl}), dtype=np.float64)
        device_grid = cfg.device_gain_grid_db if cfg.infer_device_gain else [0.0]
        self.device_gain_grid = np.array(sorted({float(x) for x in device_grid}), dtype=np.float64)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        pos_mask = x >= 0
        out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[~pos_mask])
        out[~pos_mask] = exp_x / (1.0 + exp_x)
        return out

    def _entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, _EPS, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _stats_1d(self, grid: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, list[float]]:
        mean = float(np.sum(grid * probs))
        var = float(np.sum(((grid - mean) ** 2) * probs))
        ent = self._entropy(probs)
        cdf = np.cumsum(probs)
        lo = float(grid[int(np.searchsorted(cdf, 0.025, side="left"))])
        hi = float(grid[min(int(np.searchsorted(cdf, 0.975, side="left")), len(grid) - 1)])
        return mean, var, ent, [lo, hi]

    def hearing_probability_given_theta_gain(self, amplitude_db_hl: float, theta: np.ndarray, gain: np.ndarray) -> np.ndarray:
        # Coupled reliability-aware likelihood: p(y=1|a,theta,g) with global device gain g.
        effective = float(amplitude_db_hl) + gain - theta
        clean = self._sigmoid(self.cfg.psychometric_slope * effective)
        p_heard = (1.0 - self.cfg.lapse_rate) * clean + self.cfg.lapse_rate * self.cfg.guess_rate
        return np.clip(p_heard, _EPS, 1.0 - _EPS)

    def initialize_joint_state(self, session: AudiometrySession) -> JointAudiometryState:
        theta_uniform = np.full(self.threshold_grid.shape, 1.0 / len(self.threshold_grid), dtype=np.float64)
        gain_uniform = np.full(self.device_gain_grid.shape, 1.0 / len(self.device_gain_grid), dtype=np.float64)
        g_mean, g_var, g_ent, _ = self._stats_1d(self.device_gain_grid, gain_uniform)

        state = JointAudiometryState(
            frequencies_hz=list(session.frequencies_hz),
            threshold_grid_db_hl=[float(v) for v in self.threshold_grid.tolist()],
            candidate_amplitudes_db_hl=[float(v) for v in self.candidate_amplitudes.tolist()],
            posterior_probs_by_freq={f: [float(v) for v in theta_uniform.tolist()] for f in session.frequencies_hz},
            posterior_mean_by_freq={f: float(np.mean(self.threshold_grid)) for f in session.frequencies_hz},
            posterior_variance_by_freq={f: float(np.var(self.threshold_grid)) for f in session.frequencies_hz},
            posterior_entropy_by_freq={f: float(self._entropy(theta_uniform)) for f in session.frequencies_hz},
            posterior_ci95_by_freq={f: [float(self.threshold_grid[0]), float(self.threshold_grid[-1])] for f in session.frequencies_hz},
            trial_count_by_freq={f: 0 for f in session.frequencies_hz},
            completed_by_freq={f: False for f in session.frequencies_hz},
            device_gain_grid_db=[float(v) for v in self.device_gain_grid.tolist()],
            device_gain_posterior=[float(v) for v in gain_uniform.tolist()],
            device_gain_mean=float(g_mean),
            device_gain_variance=float(g_var),
            device_gain_entropy=float(g_ent),
            shared_latents={"device_gain_db": float(g_mean)},
            shared_metadata={"gain_inference_enabled": self.cfg.infer_device_gain},
        )
        session.joint_inference_state = asdict(state)
        self._sync_session_frequency_states(session, state)
        return state

    def _sync_session_frequency_states(self, session: AudiometrySession, joint_state: JointAudiometryState) -> None:
        for f in session.frequencies_hz:
            fs = session.state_for(f)
            fs.posterior_grid_db_hl = list(joint_state.threshold_grid_db_hl)
            fs.posterior_probs = list(joint_state.posterior_probs_by_freq[f])
            fs.posterior_mean_db_hl = float(joint_state.posterior_mean_by_freq[f])
            fs.posterior_variance_db2 = float(joint_state.posterior_variance_by_freq[f])
            fs.posterior_entropy = float(joint_state.posterior_entropy_by_freq[f])
            fs.posterior_ci95 = list(joint_state.posterior_ci95_by_freq[f])
            fs.threshold_estimate_db_hl = float(joint_state.posterior_mean_by_freq[f])
            fs.uncertainty_db = float(math.sqrt(max(joint_state.posterior_variance_by_freq[f], 0.0)))
            fs.complete = bool(joint_state.completed_by_freq[f])

    def _update_gain_posterior(self, joint_state: JointAudiometryState, freq: int, amp: float, heard: bool) -> np.ndarray:
        gain_post = np.array(joint_state.device_gain_posterior, dtype=np.float64)
        theta_post = np.array(joint_state.posterior_probs_by_freq[freq], dtype=np.float64)
        theta = self.threshold_grid[:, None]
        gains = self.device_gain_grid[None, :]
        p_heard = self.hearing_probability_given_theta_gain(amp, theta, gains)
        like_g = np.sum(theta_post[:, None] * (p_heard if heard else (1.0 - p_heard)), axis=0)
        upd = gain_post * np.clip(like_g, _EPS, 1.0)
        upd /= max(float(np.sum(upd)), _EPS)
        return upd

    def _update_theta_posterior(self, joint_state: JointAudiometryState, freq: int, amp: float, heard: bool, gain_post: np.ndarray) -> np.ndarray:
        theta_post = np.array(joint_state.posterior_probs_by_freq[freq], dtype=np.float64)
        theta = self.threshold_grid[:, None]
        gains = self.device_gain_grid[None, :]
        p_heard = self.hearing_probability_given_theta_gain(amp, theta, gains)
        like_theta = np.sum(gain_post[None, :] * (p_heard if heard else (1.0 - p_heard)), axis=1)
        upd = theta_post * np.clip(like_theta, _EPS, 1.0)
        upd /= max(float(np.sum(upd)), _EPS)
        return upd

    def _apply_stats(self, joint_state: JointAudiometryState, freq: int, theta_post: np.ndarray, gain_post: np.ndarray) -> None:
        t_mean, t_var, t_ent, t_ci = self._stats_1d(self.threshold_grid, theta_post)
        joint_state.posterior_probs_by_freq[freq] = [float(v) for v in theta_post.tolist()]
        joint_state.posterior_mean_by_freq[freq] = float(t_mean)
        joint_state.posterior_variance_by_freq[freq] = float(t_var)
        joint_state.posterior_entropy_by_freq[freq] = float(t_ent)
        joint_state.posterior_ci95_by_freq[freq] = list(t_ci)

        g_mean, g_var, g_ent, _ = self._stats_1d(self.device_gain_grid, gain_post)
        joint_state.device_gain_posterior = [float(v) for v in gain_post.tolist()]
        joint_state.device_gain_mean = float(g_mean)
        joint_state.device_gain_variance = float(g_var)
        joint_state.device_gain_entropy = float(g_ent)
        joint_state.shared_latents["device_gain_db"] = float(g_mean)

    def _should_stop_frequency(self, joint_state: JointAudiometryState, freq: int) -> bool:
        trials = joint_state.trial_count_by_freq[freq]
        if trials >= self.cfg.max_trials_per_frequency:
            return True
        if trials < self.cfg.min_trials_per_frequency:
            return False
        return (
            joint_state.posterior_variance_by_freq[freq] <= self.cfg.variance_stop_threshold
            or joint_state.posterior_entropy_by_freq[freq] <= self.cfg.entropy_stop_threshold
        )

    def _expected_information_gain(self, joint_state: JointAudiometryState, freq: int, amp: float) -> float:
        theta_post = np.array(joint_state.posterior_probs_by_freq[freq], dtype=np.float64)
        gain_post = np.array(joint_state.device_gain_posterior, dtype=np.float64)
        theta = self.threshold_grid[:, None]
        gains = self.device_gain_grid[None, :]
        p_heard_mat = self.hearing_probability_given_theta_gain(amp, theta, gains)
        p_heard = float(np.sum(theta_post[:, None] * gain_post[None, :] * p_heard_mat))

        current_obj = joint_state.posterior_entropy_by_freq[freq] + 0.5 * joint_state.device_gain_entropy

        gain_yes = self._update_gain_posterior(joint_state, freq, amp, True)
        theta_yes = self._update_theta_posterior(joint_state, freq, amp, True, gain_yes)
        _, _, ent_t_yes, _ = self._stats_1d(self.threshold_grid, theta_yes)
        _, _, ent_g_yes, _ = self._stats_1d(self.device_gain_grid, gain_yes)

        gain_no = self._update_gain_posterior(joint_state, freq, amp, False)
        theta_no = self._update_theta_posterior(joint_state, freq, amp, False, gain_no)
        _, _, ent_t_no, _ = self._stats_1d(self.threshold_grid, theta_no)
        _, _, ent_g_no, _ = self._stats_1d(self.device_gain_grid, gain_no)

        expected_obj = p_heard * (ent_t_yes + 0.5 * ent_g_yes) + (1.0 - p_heard) * (ent_t_no + 0.5 * ent_g_no)
        return float(current_obj - expected_obj)

    def select_next_stimulus(self, joint_state: JointAudiometryState) -> tuple[int, float, float]:
        unfinished = [f for f in joint_state.frequencies_hz if not joint_state.completed_by_freq[f]]
        if not unfinished:
            raise RuntimeError("Joint state complete")
        selected_freq = max(unfinished, key=lambda f: (joint_state.posterior_entropy_by_freq[f], joint_state.posterior_variance_by_freq[f]))

        best_amp, best_ig = float(self.candidate_amplitudes[0]), -float("inf")
        for amp in self.candidate_amplitudes:
            ig = self._expected_information_gain(joint_state, selected_freq, float(amp))
            if ig > best_ig:
                best_amp, best_ig = float(amp), float(ig)
        return int(selected_freq), float(best_amp), float(best_ig)

    def update_joint_state(self, joint_state: JointAudiometryState, *, frequency_hz: int, amplitude_db_hl: float, heard: bool) -> JointAudiometryState:
        freq = int(frequency_hz)
        gain_post = self._update_gain_posterior(joint_state, freq, float(amplitude_db_hl), bool(heard))
        theta_post = self._update_theta_posterior(joint_state, freq, float(amplitude_db_hl), bool(heard), gain_post)
        self._apply_stats(joint_state, freq, theta_post, gain_post)

        joint_state.trial_count_by_freq[freq] += 1
        joint_state.response_history.append({"frequency_hz": freq, "amplitude_db_hl": float(amplitude_db_hl), "heard": bool(heard), "trial_at_frequency": int(joint_state.trial_count_by_freq[freq])})
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
