# Personalized Hearing-Aware Speech Enhancement

## Phase 5B2-prep: Joint Latent Audiometry Scaffold

Audiometry now runs through a **joint latent-state scaffold**:
- latent threshold vector over all 8 frequencies
- joint state container with per-frequency posteriors/summaries
- placeholder shared latent slot for future device-gain inference (`device_gain_db: None`)

Current inference still uses factorized per-frequency updates internally, but all Bayesian operations are routed through a joint API (`initialize_joint_state`, `select_next_stimulus`, `update_joint_state`, `summarize_joint_state`).

## What changed in behavior

1. Frequency selection is now joint-state-driven (entropy/variance prioritization among unfinished frequencies).
2. Amplitude selection remains one-step expected information gain.
3. Reliability-aware psychometrics (lapse/guess) are preserved.
4. Outputs remain profile-centric and backward compatible.

## Not included yet
- device-gain nuisance inference
- calibration-invariant inference
- cross-frequency GP coupling
- advanced latent human-state modeling

This phase is structural preparation for future shared latent calibration factors such as unknown device gain.

## Closed loop remains
1. joint-state Bayesian audiometry estimates thresholds
2. profile JSON stores thresholds + uncertainty + reliability
3. calibration baseline runs
4. conditioned ML personalization runs
5. listener-space comparisons on audio/video

## Example profile estimation
```bash
python -m personalized_hearing_enhancement.cli.main estimate-profile \
  --mode simulated \
  --audiometry_mode bayesian \
  --response_model lapse_logistic \
  --lapse_rate 0.10 \
  --guess_rate 0.5 \
  --simulated_audiogram "20,25,30,45,60,65,70,75"
```

## Validation
```bash
python -m personalized_hearing_enhancement.cli.main validate-audiometry \
  --audiometry_mode bayesian \
  --include_staircase_baseline \
  --runs_per_profile 5
```
