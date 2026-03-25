#!/usr/bin/env bash
set -euo pipefail

CONFIG="personalized_hearing_enhancement/configs/default.yaml"

case "${1:-}" in
  prepare_data)
    python -m personalized_hearing_enhancement.cli.main prepare-data --config "$CONFIG"
    ;;
  train_baseline)
    python -m personalized_hearing_enhancement.cli.main train --config "$CONFIG" --model-type baseline
    ;;
  train_conditioned)
    python -m personalized_hearing_enhancement.cli.main train --config "$CONFIG" --model-type conditioned
    ;;
  overfit)
    python -m personalized_hearing_enhancement.cli.main train --config "$CONFIG" --model-type conditioned --overfit_single_batch
    ;;
  debug)
    python -m personalized_hearing_enhancement.cli.main debug --config "$CONFIG"
    ;;
  demo_audio)
    python -m personalized_hearing_enhancement.cli.main demo-audio --input-wav "${2:?provide input wav}" --audiogram "${3:-20,25,30,45,60,65,70,75}"
    ;;
  process_video)
    python -m personalized_hearing_enhancement.cli.main process-video --input "${2:?provide input mp4}" --audiogram "${3:-20,25,30,45,60,65,70,75}"
    ;;
  *)
    echo "Usage: $0 {prepare_data|train_baseline|train_conditioned|overfit|debug|demo_audio <wav> [ag]|process_video <mp4> [ag]}"
    exit 1
    ;;
esac
