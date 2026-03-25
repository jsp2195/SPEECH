from __future__ import annotations

import random

import numpy as np
import torch

from personalized_hearing_enhancement.data.augment import mix_snr
from personalized_hearing_enhancement.models.tasnet import ConvTasNet
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
