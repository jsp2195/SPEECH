# Personalized Hearing-Aware Speech Enhancement

A production-ready repository for **profile-centric hearing personalization** across audio and video enhancement.

## Phase 4 Evidence Story

This repository now supports two explicit evidence claims:

1. **Output-side user-benefit comparison** using the same `HearingProfile`:
   - impaired input
   - calibration baseline (deterministic DSP baseline)
   - conditioned ML output (challenger)

2. **Front-end audiometry validation** using synthetic users with logistic yes/no psychometric responses.

Primary question:

**Does conditioned ML improve on the deterministic calibration baseline in listener space?**

## Core Personalization Contract

`HearingProfile -> hearing simulation -> calibration baseline -> conditioned ML -> metrics/reporting`

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

## Hearing Test + Profile

### Estimate hearing profile
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --simulated_audiogram "20,25,30,45,60,65,70,75" \
  --output_profile_json outputs/estimated_profile.json
```

### Show profile
```bash
python -m personalized_hearing_enhancement.cli.main show-profile \
  --profile_json outputs/estimated_profile.json \
  --save_plot_path outputs/estimated_profile.png
```

## Demo Audio (Three-way comparison)

### Preferred path: profile JSON
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio \
  --input-wav sample.wav \
  --profile_json outputs/estimated_profile.json \
  --run-name demo1
```

### Backward-compatible fallback: manual audiogram
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio \
  --input-wav sample.wav \
  --audiogram "20,25,30,45,60,65,70,75" \
  --run-name demo1
```

Precedence:
- `--profile_json` if provided
- else `--audiogram`
- else error

Outputs in `outputs/demo1/` include:
- `original.wav`
- `impaired.wav`
- `calibration.wav` (**deterministic baseline**)
- `conditioned.wav` (**ML challenger**)
- optional `baseline.wav`
- `metrics.json` with:
  - `signal_space`
  - `listener_space`
  - `hf`
  - `safety`
  - `comparison` (`conditioned_vs_calibration_*` deltas)
- `hf_energy_comparison.png`
- `safety_summary.png`

## Video Processing
```bash
python -m personalized_hearing_enhancement.cli.main process-video \
  --input sample.mp4 \
  --profile_json outputs/estimated_profile.json \
  --run-name video1
```

Outputs include `original.mp4`, `impaired.mp4`, `calibration.mp4`, `conditioned.mp4`, and `comparison_sequential.mp4`.

## Audiometry Validation (Simulated Users)

Validation uses logistic psychometric responses:

`P(heard=1 | level, threshold) = sigmoid(k * (level - threshold))`

Supported synthetic profile families:
- normal hearing
- mild loss
- sloping high-frequency loss
- irregular profile

Run validation:
```bash
python -m personalized_hearing_enhancement.cli.main validate-audiometry \
  --runs_per_profile 5 \
  --psychometric_slope 0.35 \
  --output_json outputs/audiometry_validation_summary.json
```

Summary reports include mean/median MAE, MAE by frequency, mean trials per profile, and run-level recovery errors.

## Sanity + Tests
```bash
PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_audiometry_phase2.py
PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_audiometry_validation.py
PYTHONPATH=. pytest personalized_hearing_enhancement/tests/test_determinism.py
```
