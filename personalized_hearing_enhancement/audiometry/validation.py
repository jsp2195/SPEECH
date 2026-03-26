from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from personalized_hearing_enhancement.audiometry.engine import AudiometryEngineConfig, SimulatedResponderConfig, run_hearing_test
from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ


@dataclass
class ValidationRunResult:
    profile_type: str
    scenario: str
    estimator_mode: str
    seed: int
    ground_truth_thresholds: list[float]
    estimated_thresholds: list[float]
    true_device_gain_db: float
    estimated_device_gain_db: float
    device_gain_abs_error: float
    abs_error_by_freq: dict[int, float]
    mean_abs_error: float
    total_trials: int
    trial_count_by_freq: dict[int, int]
    uncertainty_db: list[float]
    reliability_score: float | None


PROFILE_LIBRARY: dict[str, list[float]] = {
    "normal": [5, 5, 10, 10, 10, 10, 10, 10],
    "sloping_high_frequency": [10, 15, 20, 30, 45, 55, 65, 75],
    "irregular": [15, 30, 20, 45, 35, 55, 40, 60],
}

SCENARIOS: dict[str, dict] = {
    "normal_unknown_gain": {"profile_type": "normal", "true_device_gain_db": 6.0, "lapse_rate": 0.0, "inconsistency_rate": 0.0},
    "sloping_unknown_gain": {"profile_type": "sloping_high_frequency", "true_device_gain_db": -6.0, "lapse_rate": 0.0, "inconsistency_rate": 0.0},
    "irregular_unknown_gain": {"profile_type": "irregular", "true_device_gain_db": 9.0, "lapse_rate": 0.02, "inconsistency_rate": 0.02},
    "irregular_noisy_unknown_gain": {"profile_type": "irregular", "true_device_gain_db": -9.0, "lapse_rate": 0.1, "inconsistency_rate": 0.15},
}


def generate_synthetic_profile(profile_type: str, *, jitter_std: float = 0.0, seed: int | None = None) -> list[float]:
    base = np.array(PROFILE_LIBRARY[profile_type], dtype=np.float32)
    if jitter_std > 0:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0.0, jitter_std, size=base.shape)
    return [float(x) for x in np.clip(base, 0.0, 100.0).tolist()]


def run_single_validation(
    profile_type: str,
    *,
    engine_cfg: AudiometryEngineConfig,
    slope: float = 0.35,
    seed: int = 0,
    jitter_std: float = 0.0,
    scenario: str = "custom",
    true_device_gain_db: float = 0.0,
    lapse_rate: float = 0.0,
    inconsistency_rate: float = 0.0,
) -> ValidationRunResult:
    gt = generate_synthetic_profile(profile_type, jitter_std=jitter_std, seed=seed)
    responder = SimulatedResponderConfig(
        psychometric_slope=slope,
        lapse_rate=lapse_rate,
        guess_rate=0.5,
        true_device_gain_db=true_device_gain_db,
        response_model="lapse_logistic" if lapse_rate > 0 else "clean_logistic",
        inconsistency_rate=inconsistency_rate,
    )
    session, profile = run_hearing_test(
        engine_cfg,
        mode="simulated",
        ground_truth_audiogram=gt,
        seed=seed,
        simulated_responder=responder,
        verbose=False,
    )

    est = [float(v) for v in profile.thresholds_db]
    errors = [abs(a - b) for a, b in zip(gt, est, strict=True)]
    est_gain = float(profile.estimated_device_gain_db or 0.0)
    return ValidationRunResult(
        profile_type=profile_type,
        scenario=scenario,
        estimator_mode=engine_cfg.estimator_mode,
        seed=seed,
        ground_truth_thresholds=gt,
        estimated_thresholds=est,
        true_device_gain_db=float(true_device_gain_db),
        estimated_device_gain_db=est_gain,
        device_gain_abs_error=float(abs(est_gain - true_device_gain_db)),
        abs_error_by_freq={f: float(e) for f, e in zip(STANDARD_FREQS_HZ, errors, strict=True)},
        mean_abs_error=float(np.mean(errors)),
        total_trials=int(sum(len(session.state_for(f).trials) for f in session.frequencies_hz)),
        trial_count_by_freq={int(f): int(len(session.state_for(f).trials)) for f in session.frequencies_hz},
        uncertainty_db=[float(u) for u in profile.uncertainty],
        reliability_score=profile.reliability_score,
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

    by_mode: dict[str, dict] = {}
    for mode in modes:
        mode_cfg = AudiometryEngineConfig(**{**asdict(cfg), "estimator_mode": mode})
        if mode == "staircase":
            mode_cfg.infer_device_gain = False
            mode_cfg.device_gain_grid_db = [0.0]

        runs: list[ValidationRunResult] = []
        for s_idx, (name, sc) in enumerate(SCENARIOS.items()):
            for r in range(runs_per_profile):
                seed = base_seed + s_idx * 10_000 + r
                runs.append(
                    run_single_validation(
                        str(sc["profile_type"]),
                        engine_cfg=mode_cfg,
                        slope=slope,
                        seed=seed,
                        jitter_std=jitter_std,
                        scenario=name,
                        true_device_gain_db=float(sc["true_device_gain_db"]),
                        lapse_rate=float(sc["lapse_rate"]),
                        inconsistency_rate=float(sc["inconsistency_rate"]),
                    )
                )

        mae = np.array([x.mean_abs_error for x in runs], dtype=np.float32)
        gain_err = np.array([x.device_gain_abs_error for x in runs], dtype=np.float32)
        trials = np.array([x.total_trials for x in runs], dtype=np.float32)
        by_mode[mode] = {
            "mean_mae": float(np.mean(mae)),
            "mean_gain_abs_error": float(np.mean(gain_err)),
            "mean_trials_per_profile": float(np.mean(trials)),
            "mae_by_frequency": {f: float(np.mean([x.abs_error_by_freq[f] for x in runs])) for f in STANDARD_FREQS_HZ},
            "runs": [asdict(x) for x in runs],
        }

    primary = by_mode[cfg.estimator_mode]
    staircase = by_mode.get("staircase")
    return {
        "runs_per_profile": runs_per_profile,
        "scenarios": list(SCENARIOS.keys()),
        "primary_mode": cfg.estimator_mode,
        "results_by_mode": by_mode,
        "mean_mae": primary["mean_mae"],
        "mae_by_frequency": primary["mae_by_frequency"],
        "mean_trials_per_profile": primary["mean_trials_per_profile"],
        "mean_gain_abs_error": primary["mean_gain_abs_error"],
        "mae_vs_staircase_delta": (float(primary["mean_mae"] - staircase["mean_mae"]) if staircase else None),
        "gain_err_vs_staircase_delta": (float(primary["mean_gain_abs_error"] - staircase["mean_gain_abs_error"]) if staircase else None),
        "runs": primary["runs"],
    }


def save_validation_summary(summary: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out
