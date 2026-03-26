# Personalized Hearing-Aware Speech Enhancement

A production-ready repository for **profile-centric hearing personalization** across audio and video enhancement.

## Phase 5A: Bayesian Active Audiometry

The hearing-test front end now uses a **Bayesian active threshold estimator** (default mode) that keeps an explicit posterior per frequency:

1. frequency grid
2. posterior over threshold candidates
3. active amplitude selection via one-step expected information gain
4. posterior mean threshold + uncertainty summaries

The profile-driven backend remains unchanged: hearing profile estimation is still explicit and separate from DSP/ML enhancement.

### Not in this phase
- device-gain nuisance parameters
- calibration-invariant inference
- reliability-aware latent response models
- cross-frequency GP coupling (beyond future scaffolding)

## Core closed loop

1. estimate hearing profile with Bayesian active audiometry
2. save profile JSON with uncertainty
3. apply calibration baseline
4. apply conditioned ML enhancement
5. compare outputs on real audio/video

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r personalized_hearing_enhancement/requirements.txt
```

## Hearing Test + Profile

### Estimate hearing profile (Bayesian default)
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode bayesian \
  --simulated_audiogram "20,25,30,45,60,65,70,75" \
  --output_profile_json outputs/estimated_profile.json
```

### Optional baseline mode (legacy staircase)
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode staircase \
  --simulated_audiogram "20,25,30,45,60,65,70,75"
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

## Audiometry Validation (Simulated Users)

Validation uses logistic psychometric responses:

`P(heard=1 | level, threshold) = sigmoid(k * (level - threshold))`

Supported synthetic profile families:
- normal hearing
- mild loss
- sloping high-frequency loss
- irregular profile

Run validation (Bayesian with optional staircase baseline comparison):
```bash
python -m personalized_hearing_enhancement.cli.main validate-audiometry \
  --audiometry_mode bayesian \
  --include_staircase_baseline \
  --runs_per_profile 5 \
  --psychometric_slope 0.35 \
  --output_json outputs/audiometry_validation_summary.json
```

Summary includes MAE by frequency, overall MAE, trial counts, uncertainty summaries, and entropy/convergence indicators.
