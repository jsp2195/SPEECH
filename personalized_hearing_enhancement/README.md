# Personalized Hearing-Aware Speech Enhancement

A production-ready repository for **Audiogram-Conditioned Video Audio Enhancement**.
It trains baseline and conditioned Conv-TasNet speech enhancers, adds deterministic calibration filtering, and renders side-by-side demo/video comparisons.

## Processing Semantics (Important)
- **Main processing domain:** raw/noisy device audio.
- Baseline model, conditioned model, and calibration filter all process raw/noisy input directly.
- **Hearing-loss simulation is not the deployed input domain**; it is used for:
  - illustrative impaired demo artifacts,
  - listener-space evaluation metrics,
  - optional listener-aware loss during training.

## Phase 2 Explicit Pipeline
This repository now supports an explicit, interpretable two-stage flow:

1. **Hearing test / profile estimation**
   - User responses (interactive CLI) or synthetic responses (simulated mode)
   - 8-band threshold estimation on frequencies `[250, 500, 1000, 2000, 3000, 4000, 6000, 8000]`
   - Save reusable profile artifact (`.json`)
2. **Personalization stage**
   - Load profile JSON
   - Feed validated audiogram to calibration filter and/or conditioned model
   - Produce processed audio/video outputs

> Phase 2 uses explicit threshold estimation (staircase search) and is intentionally modular so it can be replaced later by Bayesian active audiometry.

## Architecture
```mermaid
flowchart LR
    A[Raw/Noisy Audio] --> B[Baseline Conv-TasNet]
    A --> C[Conditioned Conv-TasNet + Audiogram]
    A --> D[Calibration Filter + Device Profile]
    U[User hearing test] --> P[Estimated profile JSON]
    P --> E[Audiogram 8-band]
    E --> C
    E --> D
    A --> F[Original Output Artifacts]
    A --> G[Hearing-Loss Simulator H(·,θ) for Illustration + Metrics]
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r personalized_hearing_enhancement/requirements.txt
```

## Setup Data
```bash
python -m personalized_hearing_enhancement.cli.main prepare-data --config personalized_hearing_enhancement/configs/default.yaml
```

## Training
```bash
python -m personalized_hearing_enhancement.cli.main train --model-type baseline
python -m personalized_hearing_enhancement.cli.main train --model-type conditioned
```

### Debug Mode (<5 min CPU)
```bash
python -m personalized_hearing_enhancement.cli.main debug
```

### Overfit Single Batch
```bash
python -m personalized_hearing_enhancement.cli.main train --model-type conditioned --overfit_single_batch --run-name overfit_check
```

### Optional Listener-Aware Loss
Configured in `configs/default.yaml`:
- `losses.listener_space.enabled` (default `false`)
- `losses.listener_space.weight`

When enabled, training adds a listener-space term comparing `H(pred, θ)` vs `H(clean, θ)` while preserving standard signal-space loss.

## Hearing Test & Profile Commands

### Interactive hearing test (no save)
```bash
python -m personalized_hearing_enhancement.cli.main run-hearing-test --mode interactive
```

### Simulated hearing test (quick debug)
```bash
python -m personalized_hearing_enhancement.cli.main run-hearing-test --mode simulated --simulated_audiogram "20,25,30,45,60,65,70,75" --seed 7
```

### Estimate and save profile JSON
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile --mode simulated --simulated_audiogram "20,25,30,45,60,65,70,75" --output_profile_json outputs/estimated_profile.json
```

### Show profile summary (+ optional plot)
```bash
python -m personalized_hearing_enhancement.cli.main show-profile --profile_json outputs/estimated_profile.json --save_plot_path outputs/estimated_profile.png
```

## Demo Audio
### Preferred: profile JSON
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio --input-wav sample.wav --profile_json outputs/estimated_profile.json --run-name demo1
```

### Backward compatible manual audiogram
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio --input-wav sample.wav --audiogram "20,25,30,45,60,65,70,75" --run-name demo1
```

If both `--profile_json` and `--audiogram` are passed, **`--profile_json` takes precedence**.

Outputs in `outputs/demo1/` include:
- `original.wav`
- `impaired.wav` (illustrative only)
- `baseline.wav`
- `calibration.wav`
- `conditioned.wav`
- `output.wav` (selected with `--mode`)
- backward-compatible aliases: `clean.wav`, `degraded.wav`
- `waveforms.png`, `spectrograms.png`, `metrics.json`

Additional flags:
- `--mode {model,calibration}`
- `--device_profile {earbuds,headphones,airpods,overear}`
- `--max_gain_db 20`
- `--debug`

## Process Video
### Preferred: profile JSON
```bash
python -m personalized_hearing_enhancement.cli.main process-video --input sample.mp4 --profile_json outputs/estimated_profile.json --run-name video1
```

### Backward compatible manual audiogram
```bash
python -m personalized_hearing_enhancement.cli.main process-video --input sample.mp4 --audiogram "20,25,30,45,60,65,70,75" --run-name video1
```

Outputs include:
- `original.mp4`
- `impaired.mp4`
- `baseline.mp4`
- `calibration.mp4`
- `conditioned.mp4`
- `comparison_sequential.mp4` (flagship: correct per-segment audio + labels)
- `comparison_grid_visual_only.mp4` (visual-only mosaic)

## Metrics
`metrics.json` separates:
- **signal-space** metrics (e.g., SI-SDR / spectral distance on original vs output), and
- **listener-space** metrics (compare `H(original, θ)` vs `H(output, θ)`).

## Sanity and Determinism
- Hearing simulator attenuation validation at train startup.
- Identity pre-training model check at startup.
- Determinism tests: `PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_determinism.py`
- Audiometry/profile tests: `PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_audiometry_phase2.py`
