from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from personalized_hearing_enhancement.data.dataset import SpeechEnhancementDataset
from personalized_hearing_enhancement.evaluation.metrics import pesq_proxy, si_sdr, sisdr_loss, waveform_l1
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet, count_parameters
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.utils.audio import mel_spectrogram

console = Console()
app = typer.Typer()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_random_audiogram(batch: int, device: torch.device) -> torch.Tensor:
    base = torch.rand(batch, 1, device=device) * 35.0
    slope = torch.rand(batch, 1, device=device) * 8.0
    idx = torch.arange(8, device=device).view(1, -1)
    noise = torch.randn(batch, 8, device=device) * 2.0
    return torch.clamp(base + idx * slope + noise, 0.0, 90.0)


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x[0].numel() for x in batch)
    clean, noisy = [], []
    for c, n in batch:
        clean.append(torch.nn.functional.pad(c, (0, max_len - c.numel())))
        noisy.append(torch.nn.functional.pad(n, (0, max_len - n.numel())))
    return torch.stack(clean), torch.stack(noisy)


def build_model(cfg, model_type: str) -> torch.nn.Module:
    kwargs = dict(cfg.model.tasnet)
    if model_type == "baseline":
        return ConvTasNet(**kwargs)
    return ConditionedConvTasNet(**kwargs)


def compute_loss(clean: torch.Tensor, pred: torch.Tensor, sr: int) -> torch.Tensor:
    l_sisdr = sisdr_loss(clean, pred)
    l_mel = torch.mean(torch.abs(mel_spectrogram(clean, sr) - mel_spectrogram(pred, sr)))
    l_l1 = waveform_l1(clean, pred)
    return 0.7 * l_sisdr + 0.3 * l_mel + 0.1 * l_l1


def run_training(config_path: str, model_type: str) -> None:
    cfg = OmegaConf.load(config_path)
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SpeechEnhancementDataset(
        cfg.paths.data_cache,
        sr=cfg.sample_rate,
        split="train",
        segment_min_s=cfg.segment_seconds.min,
        segment_max_s=cfg.segment_seconds.max,
        snr_min_db=cfg.dataset.snr_min_db,
        snr_max_db=cfg.dataset.snr_max_db,
    )
    val_ds = SpeechEnhancementDataset(
        cfg.paths.data_cache,
        sr=cfg.sample_rate,
        split="val",
        segment_min_s=cfg.segment_seconds.min,
        segment_max_s=cfg.segment_seconds.max,
        snr_min_db=cfg.dataset.snr_min_db,
        snr_max_db=cfg.dataset.snr_max_db,
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=collate_pad)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate_pad)

    model = build_model(cfg, model_type).to(device)
    console.log(f"Model {model_type} params: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    total_steps = cfg.training.epochs * cfg.training.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - cfg.training.warmup_steps))

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_iter = iter(train_dl)
        for _ in range(cfg.training.steps_per_epoch):
            try:
                clean, noisy = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                clean, noisy = next(train_iter)
            clean, noisy = clean.to(device), noisy.to(device)
            audiogram = sample_random_audiogram(clean.size(0), device)
            degraded = apply_hearing_loss(noisy, audiogram, sr=cfg.sample_rate)

            if model_type == "conditioned":
                pred = model(degraded, audiogram)
            else:
                pred = model(degraded)

            loss = compute_loss(clean, pred, cfg.sample_rate)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            if step >= cfg.training.warmup_steps:
                scheduler.step()
            else:
                warm = float(step + 1) / float(max(1, cfg.training.warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.training.lr * warm
            step += 1

        model.eval()
        val_losses, val_sisdr = [], []
        with torch.no_grad():
            val_iter = iter(val_dl)
            for _ in range(cfg.training.val_steps):
                try:
                    clean, noisy = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dl)
                    clean, noisy = next(val_iter)
                clean, noisy = clean.to(device), noisy.to(device)
                audiogram = sample_random_audiogram(clean.size(0), device)
                degraded = apply_hearing_loss(noisy, audiogram, sr=cfg.sample_rate)
                pred = model(degraded, audiogram) if model_type == "conditioned" else model(degraded)
                vloss = compute_loss(clean, pred, cfg.sample_rate)
                val_losses.append(vloss.item())
                val_sisdr.append(si_sdr(clean, pred).mean().item())

        val_loss = float(np.mean(val_losses))
        val_sisdr_m = float(np.mean(val_sisdr))
        val_pesq = pesq_proxy(clean, pred)
        console.log(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_si_sdr={val_sisdr_m:.2f} dB, pesq~={val_pesq:.2f}")

        last_path = ckpt_dir / f"{model_type}_last.pt"
        torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, last_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / f"{model_type}_best.pt"
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, best_path)


@app.command()
def main(config: str = "personalized_hearing_enhancement/configs/default.yaml", model_type: str = "baseline") -> None:
    if model_type not in {"baseline", "conditioned"}:
        raise typer.BadParameter("model_type must be baseline or conditioned")
    run_training(config, model_type)


if __name__ == "__main__":
    app()
