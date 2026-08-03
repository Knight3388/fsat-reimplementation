"""Shared fixtures. Models are shrunk so the suite runs on CPU in reasonable time."""

import pytest
import torch

from fsat.models import RawNet3Detector

SR = 16000
TEST_DURATION = 1.0


@pytest.fixture(scope="session")
def tiny_detector():
    """A small RawNet3 with the same structure as the full model."""
    torch.manual_seed(0)
    return RawNet3Detector(
        embed_dim=32,
        channels=64,
        model_scale=4,
        sinc_filters=32,
        sinc_kernel=101,
        sinc_stride=10,
        sample_rate=SR,
    ).eval()


@pytest.fixture
def batch():
    torch.manual_seed(1)
    x = torch.randn(4, int(SR * TEST_DURATION)) * 0.1
    y = torch.tensor([0, 1, 0, 1])
    return x, y
