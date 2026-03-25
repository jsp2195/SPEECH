from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from personalized_hearing_enhancement.utils.audio import mel_spectrogram


def save_waveform_plot(waves: dict[str, torch.Tensor], path: str | Path, sr: int = 16000) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(waves)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, wave) in zip(axes, waves.items()):
        w = wave.detach().cpu().flatten()
        t = torch.arange(w.numel()) / sr
        ax.plot(t.numpy(), w.numpy(), linewidth=0.8)
        ax.set_title(name)
        ax.set_ylabel("amp")
    axes[-1].set_xlabel("sec")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)


def save_spectrogram_plot(waves: dict[str, torch.Tensor], path: str | Path, sr: int = 16000) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(waves)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, (name, wave) in enumerate(waves.items()):
        mel = mel_spectrogram(wave.unsqueeze(0), sr=sr)[0]
        axes[0, i].imshow((mel + 1e-6).log10().detach().cpu().numpy(), origin="lower", aspect="auto")
        axes[0, i].set_title(name)
        axes[0, i].set_xlabel("frames")
        axes[0, i].set_ylabel("mel bins")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)


def save_curve(values: list[float], path: str | Path, title: str, ylabel: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
