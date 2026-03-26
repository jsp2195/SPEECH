# Personalized Hearing-Aware Speech Enhancement

## Phase 5B2: Gain-aware Bayesian Audiometry

Audiometry now infers thresholds **and** a shared discrete device-gain nuisance latent.

Model:
`P(heard=1 | a, θ_f, g) = reliability_aware_sigmoid(k * (a + g - θ_f))`

The estimator maintains:
- per-frequency threshold posteriors
- shared device-gain posterior over a configurable dB grid
- uncertainty summaries for thresholds and gain

This is a constrained first step toward calibration-invariant inference.

### Not included yet
- ambient-noise nuisance parameters
- ear-coupling nuisance parameters
- GP-coupled cross-frequency priors
- full calibration-free clinical claims

## Closed loop
1. gain-aware Bayesian audiometry
2. profile save/load with threshold uncertainty and gain metadata
3. calibration baseline
4. conditioned ML enhancement
5. listener-space comparison on audio/video

## Example simulated estimation
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode bayesian \
  --infer_device_gain \
  --device_gain_grid_db "-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18" \
  --true_device_gain_db 6.0 \
  --simulated_audiogram "20,25,30,45,60,65,70,75"
```

## Validation under unknown gain
```bash
python -m personalized_hearing_enhancement.cli.main validate-audiometry \
  --audiometry_mode bayesian \
  --infer_device_gain \
  --include_staircase_baseline \
  --runs_per_profile 5
```
