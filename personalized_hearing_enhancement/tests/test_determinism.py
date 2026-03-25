from __future__ import annotations

import random

import numpy as np
import torch

from personalized_hearing_enhancement.data.augment import mix_snr
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
from personalized_hearing_enhancement.simulation.calibration_filter import apply_calibration_filter
from personalized_hearing_enhancement.evaluation.metrics import listener_space_metrics
from personalized_hearing_enhancement.simulation.hearing_loss import AUDIOGRAM_FREQS
from personalized_hearing_enhancement.utils.repro import set_global_seed


def test_model_output_determinism() -> None:
    set_global_seed(123)
    m1 = ConvTasNet()
    x = torch.randn(1, 3200)
    y1 = m1(x)

    set_global_seed(123)
    m2 = ConvTasNet()
    y2 = m2(x)

    assert torch.allclose(y1, y2, atol=1e-6)


def test_augmentation_determinism() -> None:
    set_global_seed(999)
    clean = torch.randn(1, 16000)
    noise = torch.randn(1, 16000)
    snr = random.uniform(0, 20)
    out1 = mix_snr(clean, noise, snr)

    set_global_seed(999)
    clean = torch.randn(1, 16000)
    noise = torch.randn(1, 16000)
    snr = random.uniform(0, 20)
    out2 = mix_snr(clean, noise, snr)
    assert torch.allclose(out1, out2, atol=1e-7)


def test_numpy_and_torch_seed_sync() -> None:
    set_global_seed(77)
    a1 = np.random.randn(5)
    t1 = torch.randn(5)

    set_global_seed(77)
    a2 = np.random.randn(5)
    t2 = torch.randn(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_calibration_filter_determinism() -> None:
    set_global_seed(101)
    x = torch.randn(16000)
    audiogram = torch.tensor([[20.0, 25.0, 30.0, 45.0, 60.0, 65.0, 70.0, 75.0]])
    y1 = apply_calibration_filter(x, audiogram, sample_rate=16000, device_profile="headphones", max_gain_db=20.0)

    set_global_seed(101)
    y2 = apply_calibration_filter(x, audiogram, sample_rate=16000, device_profile="headphones", max_gain_db=20.0)

    assert torch.allclose(y1, y2, atol=1e-6)



def test_audiogram_grid_within_nyquist() -> None:
    assert AUDIOGRAM_FREQS.numel() == 8
    assert float(AUDIOGRAM_FREQS.max().item()) <= 8000.0


def test_listener_space_metrics_shapes() -> None:
    clean = torch.randn(16000)
    estimate = torch.randn(16000)
    audiogram = torch.tensor([[20.0, 25.0, 30.0, 45.0, 55.0, 60.0, 65.0, 70.0]])
    metrics = listener_space_metrics(clean, estimate, audiogram, sr=16000)
    assert "listener_si_sdr" in metrics
    assert "listener_log_spectral_distance" in metrics



def test_conditioned_model_accepts_tasnet_config_keys() -> None:
    from personalized_hearing_enhancement.models.conditioned_tasnet import ConditionedConvTasNet

    model = ConditionedConvTasNet(
        encoder_dim=192,
        feature_dim=96,
        hidden_dim=384,
        kernel_size=16,
        tcn_layers=2,
        tcn_stacks=1,
        bottleneck_dim=128,
    )
    x = torch.randn(2, 3200)
    ag = torch.randn(2, 8)
    y = model(x, ag)
    assert y.shape == x.shape
