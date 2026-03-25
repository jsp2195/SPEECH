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

## Architecture
```mermaid
flowchart LR
    A[Raw/Noisy Audio] --> B[Baseline Conv-TasNet]
    A --> C[Conditioned Conv-TasNet + Audiogram]
    A --> D[Calibration Filter + Device Profile]
    E[Audiogram 8-band] --> C
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

## Demo Audio
```bash
python -m personalized_hearing_enhancement.cli.main demo-audio --input-wav sample.wav --audiogram "20,25,30,45,60,65,70,75" --run-name demo1
```

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
