from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import (
    AudiometryEngineConfig,
    SimulatedResponderConfig,
    logistic_hear_probability,
    run_hearing_test,
)
from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ


@dataclass
class ValidationRunResult:
    profile_type: str
    estimator_mode: str
    seed: int
    ground_truth_thresholds: list[float]
    estimated_thresholds: list[float]
    abs_error_by_freq: dict[int, float]
    mean_abs_error: float
    total_trials: int
    uncertainty_db: list[float]
    posterior_entropy: list[float]


PROFILE_LIBRARY: dict[str, list[float]] = {
    "normal": [5, 5, 10, 10, 10, 10, 10, 10],
    "mild_loss": [15, 20, 20, 25, 25, 30, 30, 35],
    "sloping_high_frequency": [10, 15, 20, 30, 45, 55, 65, 75],
    "irregular": [15, 30, 20, 45, 35, 55, 40, 60],
}


def generate_synthetic_profile(profile_type: str, *, jitter_std: float = 0.0, seed: int | None = None) -> list[float]:
    if profile_type not in PROFILE_LIBRARY:
        raise ValueError(f"Unknown profile_type: {profile_type}")
    base = np.array(PROFILE_LIBRARY[profile_type], dtype=np.float32)
    if jitter_std > 0:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0.0, jitter_std, size=base.shape)
    base = np.clip(base, 0.0, 100.0)
    return [float(x) for x in base.tolist()]


def make_logistic_response_callback(
    ground_truth_thresholds: list[float],
    *,
    slope: float = 0.35,
    seed: int | None = None,
):
    rng = random.Random(seed)

    def _callback(frequency_hz: int, amplitude_db_hl: float, _trial_idx: int) -> bool:
        idx = STANDARD_FREQS_HZ.index(frequency_hz)
        threshold = ground_truth_thresholds[idx]
        p_heard = logistic_hear_probability(amplitude_db_hl, threshold, slope=slope)
        return bool(rng.random() < p_heard)

    return _callback


def run_single_validation(
    profile_type: str,
    *,
    engine_cfg: AudiometryEngineConfig,
    slope: float = 0.35,
    seed: int = 0,
    jitter_std: float = 0.0,
) -> ValidationRunResult:
    gt = generate_synthetic_profile(profile_type, jitter_std=jitter_std, seed=seed)
    callback = make_logistic_response_callback(gt, slope=slope, seed=seed)
    session, estimated_profile = run_hearing_test(
        engine_cfg,
        mode="simulated",
        ground_truth_audiogram=gt,
        seed=seed,
        simulated_responder=SimulatedResponderConfig(psychometric_slope=slope),
        response_callback=callback,
        verbose=False,
    )

    estimated = [float(v) for v in estimated_profile.thresholds_db]
    errors = [abs(a - b) for a, b in zip(gt, estimated, strict=True)]
    abs_error_by_freq = {freq: float(err) for freq, err in zip(STANDARD_FREQS_HZ, errors, strict=True)}
    total_trials = int(sum(len(session.state_for(freq).trials) for freq in session.frequencies_hz))
    uncertainty = [float(u) for u in estimated_profile.uncertainty]

    entropy_values: list[float] = []
    if estimated_profile.posterior_entropy:
        entropy_values = [float(v) for v in estimated_profile.posterior_entropy]

    return ValidationRunResult(
        profile_type=profile_type,
        estimator_mode=engine_cfg.estimator_mode,
        seed=seed,
        ground_truth_thresholds=gt,
        estimated_thresholds=estimated,
        abs_error_by_freq=abs_error_by_freq,
        mean_abs_error=float(np.mean(errors)),
        total_trials=total_trials,
        uncertainty_db=uncertainty,
        posterior_entropy=entropy_values,
    )


def run_validation_suite(
    *,
    runs_per_profile: int = 5,
    slope: float = 0.35,
    base_seed: int = 0,
    jitter_std: float = 1.5,
    engine_cfg: AudiometryEngineConfig | None = None,
    include_staircase_baseline: bool = True,
) -> dict:
    cfg = engine_cfg or AudiometryEngineConfig()
    modes = [cfg.estimator_mode]
    if include_staircase_baseline and cfg.estimator_mode != "staircase":
        modes.append("staircase")

    mode_summaries: dict[str, dict] = {}
    for mode in modes:
        mode_cfg = AudiometryEngineConfig(**{**asdict(cfg), "estimator_mode": mode})
        results: list[ValidationRunResult] = []

        for profile_idx, profile_type in enumerate(PROFILE_LIBRARY.keys()):
            for run_idx in range(runs_per_profile):
                seed = base_seed + profile_idx * 10_000 + run_idx
                results.append(
                    run_single_validation(
                        profile_type,
                        engine_cfg=mode_cfg,
                        slope=slope,
                        seed=seed,
                        jitter_std=jitter_std,
                    )
                )

        mae_values = np.array([r.mean_abs_error for r in results], dtype=np.float32)
        trial_values = np.array([r.total_trials for r in results], dtype=np.float32)
        uncertainty_values = np.array([np.mean(r.uncertainty_db) for r in results], dtype=np.float32)
        mae_by_frequency: dict[int, float] = {}
        for freq in STANDARD_FREQS_HZ:
            mae_by_frequency[freq] = float(np.mean([r.abs_error_by_freq[freq] for r in results]))

        entropy_mean = float("nan")
        entropy_delta = float("nan")
        all_entropies = [np.mean(r.posterior_entropy) for r in results if r.posterior_entropy]
        if all_entropies:
            entropy_mean = float(np.mean(all_entropies))
            initial_entropy = float(np.log(len(mode_cfg.candidate_amplitudes_db_hl) + 1))
            entropy_delta = float(initial_entropy - entropy_mean)

        mode_summaries[mode] = {
            "mean_mae": float(np.mean(mae_values)),
            "median_mae": float(np.median(mae_values)),
            "mae_by_frequency": mae_by_frequency,
            "mean_trials_per_profile": float(np.mean(trial_values)),
            "mean_uncertainty_db": float(np.mean(uncertainty_values)),
            "mean_posterior_entropy": entropy_mean,
            "mean_entropy_reduction": entropy_delta,
            "error_distribution": [float(x) for x in mae_values.tolist()],
            "runs": [asdict(r) for r in results],
        }

    summary = {
        "runs_per_profile": runs_per_profile,
        "profile_types": list(PROFILE_LIBRARY.keys()),
        "psychometric_slope": slope,
        "estimator_modes": modes,
        "primary_mode": cfg.estimator_mode,
        "results_by_mode": mode_summaries,
    }
    primary = mode_summaries[cfg.estimator_mode]
    summary.update(
        {
            "mean_mae": primary["mean_mae"],
            "median_mae": primary["median_mae"],
            "mae_by_frequency": primary["mae_by_frequency"],
            "mean_trials_per_profile": primary["mean_trials_per_profile"],
            "error_distribution": primary["error_distribution"],
            "runs": primary["runs"],
        }
    )
    return summary


def save_validation_summary(summary: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out
