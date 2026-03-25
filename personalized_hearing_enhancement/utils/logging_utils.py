from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


def build_logger(run_dir: Path, name: str = "phe") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    return logger


def log_json(run_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
