# Personalized Hearing-Aware Speech Enhancement

A production-ready repository for **profile-centric hearing personalization** across audio and video enhancement.

## Phase 5B1: Reliability-Aware Bayesian Audiometry

The hearing-test front end now uses **reliability-aware Bayesian active audiometry**:

1. Bayesian threshold posterior on a discrete dB grid
2. likelihood with lapse/guess behavior
3. one-step expected information gain probe selection
4. posterior mean + uncertainty + reliability summary

Psychometric model:

`P(heard=1 | a, θ, λ, g) = (1-λ)*sigmoid(k*(a-θ)) + λ*g`

Where `λ` is lapse rate and `g` is guess rate.

### Not included yet
- device-gain nuisance parameters
- calibration-invariant inference
- shared latent hardware gain
- full latent human-state models
- cross-frequency GP coupling

## Closed loop

1. reliability-aware Bayesian audiometry estimates profile
2. profile JSON is saved with uncertainty + reliability metadata
3. calibration baseline is applied
4. conditioned ML personalization is applied
5. listener-space comparisons run on audio/video outputs

## Hearing Test + Profile

### Estimate profile (simulated, reliability-aware)
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode bayesian \
  --response_model lapse_logistic \
  --lapse_rate 0.10 \
  --guess_rate 0.5 \
  --simulate_fatigue \
  --fatigue_lapse_increment 0.006 \
  --inconsistency_rate 0.10 \
  --simulated_audiogram "20,25,30,45,60,65,70,75" \
  --output_profile_json outputs/estimated_profile.json
```

### Legacy baseline mode (optional)
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode staircase \
  --simulated_audiogram "20,25,30,45,60,65,70,75"
```

## Validation (clean + noisy responder scenarios)

`validate-audiometry` runs scenario coverage including:
- normal clean
- normal lapse-noise
- sloping HF clean
- sloping HF lapse-noise
- irregular inconsistent responder
- mild-loss fatigue responder

```bash
python -m personalized_hearing_enhancement.cli.main validate-audiometry \
  --audiometry_mode bayesian \
  --include_staircase_baseline \
  --runs_per_profile 5 \
  --psychometric_slope 0.35 \
  --output_json outputs/audiometry_validation_summary.json
```

Summary includes MAE, trials, uncertainty, reliability score, and scenario-level comparisons.
