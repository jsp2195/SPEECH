from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from personalized_hearing_enhancement.data.dataset import SpeechEnhancementDataset
from personalized_hearing_enhancement.data.download import download_datasets
from personalized_hearing_enhancement.evaluation.metrics import pesq_proxy, si_sdr, sisdr_loss, waveform_l1
from personalized_hearing_enhancement.evaluation.sanity_checks import hearing_simulator_validation, identity_model_check
from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet
from personalized_hearing_enhancement.models.tasnet import ConvTasNet, count_parameters
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.utils.audio import mel_spectrogram
from personalized_hearing_enhancement.utils.logging_utils import build_logger, log_json
from personalized_hearing_enhancement.utils.plotting import save_curve
from personalized_hearing_enhancement.utils.repro import seed_worker, set_global_seed

app = typer.Typer()


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


def run_training(
    config_path: str,
    model_type: str,
    debug: bool = False,
    overfit_single_batch: bool = False,
    run_name: str = "train",
) -> Path:
    cfg = OmegaConf.load(config_path)
    set_global_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.paths.outputs) / run_name
    logger = build_logger(run_dir, name=f"phe_train_{run_name}")
    logger.info(f"Device: {device}")

    try:
        train_ds = SpeechEnhancementDataset(
            cfg.paths.data_cache,
            sr=cfg.sample_rate,
            split="train",
            segment_min_s=cfg.segment_seconds.min,
            segment_max_s=cfg.segment_seconds.max,
            snr_min_db=cfg.dataset.snr_min_db,
            snr_max_db=cfg.dataset.snr_max_db,
            max_samples=cfg.debug.max_samples if debug else None,
        )
        val_ds = SpeechEnhancementDataset(
            cfg.paths.data_cache,
            sr=cfg.sample_rate,
            split="val",
            segment_min_s=cfg.segment_seconds.min,
            segment_max_s=cfg.segment_seconds.max,
            snr_min_db=cfg.dataset.snr_min_db,
            snr_max_db=cfg.dataset.snr_max_db,
            max_samples=cfg.debug.max_samples if debug else None,
        )
    except FileNotFoundError:
        logger.warning("Datasets missing; attempting auto-download.")
        download_datasets(cfg.paths.data_cache)
        train_ds = SpeechEnhancementDataset(cfg.paths.data_cache, sr=cfg.sample_rate, split="train", max_samples=cfg.debug.max_samples if debug else None)
        val_ds = SpeechEnhancementDataset(cfg.paths.data_cache, sr=cfg.sample_rate, split="val", max_samples=cfg.debug.max_samples if debug else None)

    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_pad,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_pad,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_model(cfg, model_type).to(device)
    logger.info(f"Model {model_type} params: {count_parameters(model):,}")

    # Sanity checks
    sim_check = hearing_simulator_validation(sr=cfg.sample_rate)
    id_check = identity_model_check(model)
    logger.info(f"hearing_simulator_validation: {sim_check}")
    logger.info(f"identity_model_check: {id_check}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    phase_epochs = [cfg.training.epochs]
    phase_settings = [{"hearing_loss": True, "conditioned": model_type == "conditioned"}]
    if bool(cfg.curriculum.enabled):
        phase_epochs = [int(cfg.curriculum.phase1.epochs), int(cfg.curriculum.phase2.epochs)]
        phase_settings = [
            {"hearing_loss": bool(cfg.curriculum.phase1.hearing_loss), "conditioned": bool(cfg.curriculum.phase1.conditioned)},
            {"hearing_loss": bool(cfg.curriculum.phase2.hearing_loss), "conditioned": bool(cfg.curriculum.phase2.conditioned)},
        ]

    steps_per_epoch = int(cfg.debug.train_steps) if debug else int(cfg.training.steps_per_epoch)
    val_steps = int(cfg.debug.val_steps) if debug else int(cfg.training.val_steps)
    if overfit_single_batch:
        steps_per_epoch = 200
        val_steps = 1

    total_steps = max(1, sum(phase_epochs) * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - int(cfg.training.warmup_steps)))

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    train_losses: list[float] = []
    val_losses_all: list[float] = []

    overfit_batch = next(iter(train_dl)) if overfit_single_batch else None

    for phase_idx, (epochs, phase) in enumerate(zip(phase_epochs, phase_settings), start=1):
        for epoch in range(epochs):
            model.train()
            train_iter = iter(train_dl)
            for _ in range(steps_per_epoch):
                if overfit_single_batch and overfit_batch is not None:
                    clean, noisy = overfit_batch
                else:
                    try:
                        clean, noisy = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_dl)
                        clean, noisy = next(train_iter)

                clean, noisy = clean.to(device), noisy.to(device)
                audiogram = sample_random_audiogram(clean.size(0), device)
                inp = apply_hearing_loss(noisy, audiogram, sr=cfg.sample_rate) if phase["hearing_loss"] else noisy

                use_conditioning = phase["conditioned"] and model_type == "conditioned"
                pred = model(inp, audiogram) if use_conditioning else model(inp)

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
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses, val_sisdr = [], []
            with torch.no_grad():
                val_iter = iter(val_dl)
                for _ in range(val_steps):
                    try:
                        clean, noisy = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dl)
                        clean, noisy = next(val_iter)
                    clean, noisy = clean.to(device), noisy.to(device)
                    audiogram = sample_random_audiogram(clean.size(0), device)
                    inp = apply_hearing_loss(noisy, audiogram, sr=cfg.sample_rate) if phase["hearing_loss"] else noisy
                    use_conditioning = phase["conditioned"] and model_type == "conditioned"
                    pred = model(inp, audiogram) if use_conditioning else model(inp)
                    vloss = compute_loss(clean, pred, cfg.sample_rate)
                    val_losses.append(vloss.item())
                    val_sisdr.append(si_sdr(clean, pred).mean().item())

            val_loss = float(np.mean(val_losses))
            val_sisdr_m = float(np.mean(val_sisdr))
            val_pesq = pesq_proxy(clean, pred)
            val_losses_all.append(val_loss)
            logger.info(
                f"phase={phase_idx} epoch={epoch+1}/{epochs} val_loss={val_loss:.4f}, val_si_sdr={val_sisdr_m:.2f} dB, pesq~={val_pesq:.2f}"
            )

            last_path = ckpt_dir / f"{model_type}_last.pt"
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, last_path)
            if val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / f"{model_type}_best.pt"
                torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, best_path)

    save_curve(train_losses, run_dir / "train_loss_curve.png", "Train Loss", "loss")
    save_curve(val_losses_all, run_dir / "val_loss_curve.png", "Validation Loss", "loss")
    log_json(
        run_dir,
        "metrics.json",
        {
            "train_loss_final": train_losses[-1] if train_losses else None,
            "val_loss_best": best_val,
            "overfit_single_batch": overfit_single_batch,
            "debug": debug,
            "sanity": {"hearing_simulator": sim_check.details, "identity_model": id_check.details},
        },
    )
    return run_dir


@app.command()
def main(
    config: str = "personalized_hearing_enhancement/configs/default.yaml",
    model_type: str = "baseline",
    debug: bool = False,
    overfit_single_batch: bool = False,
    run_name: str = "train",
) -> None:
    if model_type not in {"baseline", "conditioned"}:
        raise typer.BadParameter("model_type must be baseline or conditioned")
    run_training(config, model_type, debug=debug, overfit_single_batch=overfit_single_batch, run_name=run_name)


if __name__ == "__main__":
    app()
