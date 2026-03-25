# Personalized Hearing-Aware Speech Enhancement

A production-ready repository for **profile-centric hearing personalization** across audio and video enhancement.

## Phase 3 Personalization Contract

This repository now uses a single validated `HearingProfile` object as the primary internal contract:

`HearingProfile -> hearing simulation -> calibration baseline -> conditioned ML -> evaluation outputs`

### Baseline and challenger semantics
- **Calibration** is the first-class, deterministic, interpretable baseline.
- **Conditioned ML** is the profile-conditioned challenger.
- The main question reported by demos/evaluation is: **“Does conditioned ML beat the calibration baseline?”**

## Processing Semantics
- Main processing domain remains raw/noisy device audio.
- Baseline model, conditioned model, and calibration all process the same raw input.
- Hearing-loss simulation is used for illustrative impaired outputs and listener-space evaluation.

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

## Hearing Test and Profile

### 1) Estimate hearing profile
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --simulated_audiogram "20,25,30,45,60,65,70,75" \
  --output_profile_json outputs/estimated_profile.json
```

### 2) Inspect saved profile JSON
```bash
python -m personalized_hearing_enhancement.cli.main show-profile \
  --profile_json outputs/estimated_profile.json \
  --save_plot_path outputs/estimated_profile.png
```

## Demo Audio (profile_json preferred)

### Preferred path: `--profile_json`
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio \
  --input-wav sample.wav \
  --profile_json outputs/estimated_profile.json \
  --run-name demo1
```

### Backward-compatible fallback: `--audiogram`
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio \
  --input-wav sample.wav \
  --audiogram "20,25,30,45,60,65,70,75" \
  --run-name demo1
```

Precedence rule:
- If `--profile_json` is provided, it is used.
- Else if `--audiogram` is provided, a temporary manual `HearingProfile` is created.
- Else the command errors clearly.

Outputs (`outputs/demo1/`) are consistently ordered and labeled:
1. `original.wav`
2. `impaired.wav` (illustrative listener preview)
3. `calibration.wav` (**baseline**)
4. `baseline.wav` (unconditioned model)
5. `conditioned.wav` (profile-conditioned ML)

Also saved:
- `output.wav` (`--mode {model,calibration}`)
- `metrics.json` with `signal_space`, `listener_space`, and explicit calibration-vs-conditioned deltas.

## Process Video (profile_json preferred)

### Preferred path: `--profile_json`
```bash
python -m personalized_hearing_enhancement.cli.main process-video \
  --input sample.mp4 \
  --profile_json outputs/estimated_profile.json \
  --run-name video1
```

### Backward-compatible fallback: `--audiogram`
```bash
python -m personalized_hearing_enhancement.cli.main process-video \
  --input sample.mp4 \
  --audiogram "20,25,30,45,60,65,70,75" \
  --run-name video1
```

Outputs include:
- `original.mp4`
- `impaired.mp4`
- `baseline.mp4`
- `calibration.mp4` (**baseline**)
- `conditioned.mp4` (ML challenger)
- `comparison_sequential.mp4`
- `comparison_grid_visual_only.mp4`

## Sanity and Determinism
- `PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_determinism.py`
- `PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_audiometry_phase2.py`
