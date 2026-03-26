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
    scenario: str
    estimator_mode: str
    seed: int
    ground_truth_thresholds: list[float]
    estimated_thresholds: list[float]
    abs_error_by_freq: dict[int, float]
    mean_abs_error: float
    total_trials: int
    uncertainty_db: list[float]
    posterior_entropy: list[float]
    reliability_score: float | None
    trial_count_by_freq: dict[int, int]


PROFILE_LIBRARY: dict[str, list[float]] = {
    "normal": [5, 5, 10, 10, 10, 10, 10, 10],
    "mild_loss": [15, 20, 20, 25, 25, 30, 30, 35],
    "sloping_high_frequency": [10, 15, 20, 30, 45, 55, 65, 75],
    "irregular": [15, 30, 20, 45, 35, 55, 40, 60],
}

SCENARIOS: dict[str, dict] = {
    "normal_clean": {"profile_type": "normal", "response_model": "clean_logistic", "lapse_rate": 0.0, "guess_rate": 0.5, "inconsistency_rate": 0.0, "simulate_fatigue": False, "fatigue_lapse_increment": 0.0},
    "normal_lapse": {"profile_type": "normal", "response_model": "lapse_logistic", "lapse_rate": 0.1, "guess_rate": 0.5, "inconsistency_rate": 0.0, "simulate_fatigue": False, "fatigue_lapse_increment": 0.0},
    "sloping_clean": {"profile_type": "sloping_high_frequency", "response_model": "clean_logistic", "lapse_rate": 0.0, "guess_rate": 0.5, "inconsistency_rate": 0.0, "simulate_fatigue": False, "fatigue_lapse_increment": 0.0},
    "sloping_lapse": {"profile_type": "sloping_high_frequency", "response_model": "lapse_logistic", "lapse_rate": 0.12, "guess_rate": 0.5, "inconsistency_rate": 0.0, "simulate_fatigue": False, "fatigue_lapse_increment": 0.0},
    "irregular_inconsistent": {"profile_type": "irregular", "response_model": "lapse_logistic", "lapse_rate": 0.08, "guess_rate": 0.5, "inconsistency_rate": 0.18, "simulate_fatigue": False, "fatigue_lapse_increment": 0.0},
    "mild_fatigue": {"profile_type": "mild_loss", "response_model": "lapse_logistic", "lapse_rate": 0.06, "guess_rate": 0.5, "inconsistency_rate": 0.05, "simulate_fatigue": True, "fatigue_lapse_increment": 0.006},
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
    scenario: str = "custom",
    simulated_responder: SimulatedResponderConfig | None = None,
) -> ValidationRunResult:
    gt = generate_synthetic_profile(profile_type, jitter_std=jitter_std, seed=seed)
    responder = simulated_responder or SimulatedResponderConfig(psychometric_slope=slope)

    session, estimated_profile = run_hearing_test(
        engine_cfg,
        mode="simulated",
        ground_truth_audiogram=gt,
        seed=seed,
        simulated_responder=responder,
        response_callback=None,
        verbose=False,
    )

    estimated = [float(v) for v in estimated_profile.thresholds_db]
    errors = [abs(a - b) for a, b in zip(gt, estimated, strict=True)]
    abs_error_by_freq = {freq: float(err) for freq, err in zip(STANDARD_FREQS_HZ, errors, strict=True)}
    trial_count_by_freq = {int(freq): int(len(session.state_for(freq).trials)) for freq in session.frequencies_hz}
    total_trials = int(sum(trial_count_by_freq.values()))
    uncertainty = [float(u) for u in estimated_profile.uncertainty]

    entropy_values: list[float] = []
    if estimated_profile.posterior_entropy:
        entropy_values = [float(v) for v in estimated_profile.posterior_entropy]

    return ValidationRunResult(
        profile_type=profile_type,
        scenario=scenario,
        estimator_mode=engine_cfg.estimator_mode,
        seed=seed,
        ground_truth_thresholds=gt,
        estimated_thresholds=estimated,
        abs_error_by_freq=abs_error_by_freq,
        mean_abs_error=float(np.mean(errors)),
        total_trials=total_trials,
        uncertainty_db=uncertainty,
        posterior_entropy=entropy_values,
        reliability_score=estimated_profile.reliability_score,
        trial_count_by_freq=trial_count_by_freq,
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

        for scenario_idx, (scenario_name, scenario_cfg) in enumerate(SCENARIOS.items()):
            for run_idx in range(runs_per_profile):
                seed = base_seed + scenario_idx * 10_000 + run_idx
                responder = SimulatedResponderConfig(
                    psychometric_slope=slope,
                    lapse_rate=float(scenario_cfg["lapse_rate"]),
                    guess_rate=float(scenario_cfg["guess_rate"]),
                    response_model=str(scenario_cfg["response_model"]),
                    inconsistency_rate=float(scenario_cfg["inconsistency_rate"]),
                    simulate_fatigue=bool(scenario_cfg["simulate_fatigue"]),
                    fatigue_lapse_increment=float(scenario_cfg["fatigue_lapse_increment"]),
                )
                scenario_engine_cfg = AudiometryEngineConfig(
                    **{
                        **asdict(mode_cfg),
                        "lapse_rate": responder.lapse_rate,
                        "guess_rate": responder.guess_rate,
                    }
                )
                results.append(
                    run_single_validation(
                        str(scenario_cfg["profile_type"]),
                        engine_cfg=scenario_engine_cfg,
                        slope=slope,
                        seed=seed,
                        jitter_std=jitter_std,
                        scenario=scenario_name,
                        simulated_responder=responder,
                    )
                )

        mae_values = np.array([r.mean_abs_error for r in results], dtype=np.float32)
        trial_values = np.array([r.total_trials for r in results], dtype=np.float32)
        uncertainty_values = np.array([np.mean(r.uncertainty_db) for r in results], dtype=np.float32)
        reliability_values = np.array([r.reliability_score if r.reliability_score is not None else np.nan for r in results], dtype=np.float32)

        mae_by_frequency: dict[int, float] = {}
        for freq in STANDARD_FREQS_HZ:
            mae_by_frequency[freq] = float(np.mean([r.abs_error_by_freq[freq] for r in results]))

        scenario_summary: dict[str, dict[str, float]] = {}
        for scenario_name in SCENARIOS:
            scenario_runs = [r for r in results if r.scenario == scenario_name]
            scenario_summary[scenario_name] = {
                "mean_mae": float(np.mean([r.mean_abs_error for r in scenario_runs])),
                "mean_trials": float(np.mean([r.total_trials for r in scenario_runs])),
                "mean_reliability": float(np.nanmean([r.reliability_score for r in scenario_runs])),
            }

        mode_summaries[mode] = {
            "mean_mae": float(np.mean(mae_values)),
            "median_mae": float(np.median(mae_values)),
            "mae_by_frequency": mae_by_frequency,
            "mean_trials_per_profile": float(np.mean(trial_values)),
            "mean_uncertainty_db": float(np.mean(uncertainty_values)),
            "mean_reliability_score": float(np.nanmean(reliability_values)),
            "scenario_summary": scenario_summary,
            "error_distribution": [float(x) for x in mae_values.tolist()],
            "runs": [asdict(r) for r in results],
        }

    summary = {
        "runs_per_profile": runs_per_profile,
        "psychometric_slope": slope,
        "estimator_modes": modes,
        "primary_mode": cfg.estimator_mode,
        "scenarios": list(SCENARIOS.keys()),
        "results_by_mode": mode_summaries,
    }
    primary = mode_summaries[cfg.estimator_mode]
    staircase = mode_summaries.get("staircase")
    summary.update(
        {
            "mean_mae": primary["mean_mae"],
            "median_mae": primary["median_mae"],
            "mae_by_frequency": primary["mae_by_frequency"],
            "mean_trials_per_profile": primary["mean_trials_per_profile"],
            "mean_reliability_score": primary["mean_reliability_score"],
            "error_distribution": primary["error_distribution"],
            "runs": primary["runs"],
            "scenario_summary": primary["scenario_summary"],
            "mae_vs_staircase_delta": (float(primary["mean_mae"] - staircase["mean_mae"]) if staircase else None),
        }
    )
    return summary


def save_validation_summary(summary: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out
