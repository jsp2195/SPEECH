from __future__ import annotations

import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from personalized_hearing_enhancement.data.augment import apply_rir, mix_snr, random_crop_or_pad


class SpeechEnhancementDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_cache: str | Path,
        sr: int = 16000,
        split: str = "train",
        segment_min_s: float = 2.0,
        segment_max_s: float = 5.0,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(data_cache)
        self.sr = sr
        self.segment_min_s = segment_min_s
        self.segment_max_s = segment_max_s
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.max_samples = max_samples

        libri_subdir = "train-clean-100" if split == "train" else "dev-clean"
        self.clean_files = sorted((self.root / "LibriSpeech" / libri_subdir).rglob("*.flac"))
        self.noise_files = sorted((self.root / "musan").rglob("*.wav"))
        self.rir_files = sorted((self.root / "RIRS_NOISES").rglob("*.wav"))

        if not self.clean_files:
            raise FileNotFoundError("No LibriSpeech files found. Run data/download.py first.")
        if not self.noise_files:
            raise FileNotFoundError("No MUSAN files found. Run data/download.py first.")
        if not self.rir_files:
            raise FileNotFoundError("No RIR files found. Run data/download.py first.")

    def __len__(self) -> int:
        n = max(10_000, len(self.clean_files))
        return min(n, self.max_samples) if self.max_samples is not None else n

    def _load_mono_resampled(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path.as_posix())
        wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.clean_files[idx % len(self.clean_files)]
        noise_path = random.choice(self.noise_files)
        rir_path = random.choice(self.rir_files)

        seg_len = int(random.uniform(self.segment_min_s, self.segment_max_s) * self.sr)

        clean = random_crop_or_pad(self._load_mono_resampled(clean_path), seg_len)
        noise = random_crop_or_pad(self._load_mono_resampled(noise_path), seg_len)

        rir = self._load_mono_resampled(rir_path).squeeze(0)
        if rir.numel() > self.sr * 2:
            rir = rir[: self.sr * 2]

        noisy = mix_snr(clean, noise, random.uniform(self.snr_min_db, self.snr_max_db))
        noisy = apply_rir(noisy, rir)

        peak = torch.clamp(noisy.abs().max(), min=1e-6)
        noisy = torch.clamp(noisy / peak, -1.0, 1.0)
        clean = torch.clamp(clean / torch.clamp(clean.abs().max(), min=1e-6), -1.0, 1.0)

        return clean.squeeze(0), noisy.squeeze(0)
