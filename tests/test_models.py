"""Tests for the RawNet3 backbone and detector head."""

import numpy as np
import pytest
import torch

from fsat.models import AFMS, Bottle2neck, ParamSincFB, PreEmphasis, RawNet3, RawNet3Detector

SR = 16000


def test_detector_output_shape(tiny_detector, batch):
    x, _ = batch
    assert tiny_detector(x).shape == (x.size(0), 2)


def test_detector_accepts_channel_dimension(tiny_detector, batch):
    x, _ = batch
    assert torch.allclose(tiny_detector(x.unsqueeze(1)), tiny_detector(x), atol=1e-5)


def test_short_input_raises_a_clear_error(tiny_detector):
    with pytest.raises(ValueError, match="too short"):
        tiny_detector(torch.randn(2, 50))


def test_minimum_length_input_is_accepted(tiny_detector):
    x = torch.randn(2, tiny_detector.min_input_samples) * 0.1
    assert tiny_detector(x).shape == (2, 2)


def test_gradients_reach_the_sinc_filterbank(tiny_detector, batch):
    x, y = batch
    tiny_detector.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(tiny_detector(x), y).backward()
    assert tiny_detector.backbone.conv1.low_hz_.grad is not None
    assert torch.isfinite(tiny_detector.backbone.conv1.low_hz_.grad).all()
    tiny_detector.zero_grad(set_to_none=True)


def test_gradient_flows_to_the_input(tiny_detector, batch):
    """Required for any input-space attack to work at all."""
    x, y = batch
    x = x.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(tiny_detector(x), y).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---------------------------------------------------------------- ParamSincFB
def test_sinc_filterbank_shape():
    fb = ParamSincFB(n_filters=32, kernel_size=101, stride=10, sample_rate=SR)
    filters = fb.make_filters()
    assert filters.shape == (32, 1, 101)
    assert torch.isfinite(filters).all()


def test_sinc_filterbank_produces_cos_sin_pairs():
    """The analytic bank is n_filters//2 cosine kernels then the sine partners."""
    fb = ParamSincFB(n_filters=16, kernel_size=51, sample_rate=SR)
    filters = fb.make_filters().detach()[:, 0, :]
    cos_part, sin_part = filters[:8], filters[8:]
    centre = 51 // 2
    # Cosine kernels are even about the centre, sine kernels are odd.
    assert torch.allclose(cos_part, torch.flip(cos_part, dims=[1]), atol=1e-5)
    assert torch.allclose(sin_part, -torch.flip(sin_part, dims=[1]), atol=1e-5)
    assert float(sin_part[:, centre].abs().max()) < 1e-6


def test_sinc_filters_are_bandpass():
    """Each learned filter must concentrate energy in its own band."""
    fb = ParamSincFB(n_filters=32, kernel_size=251, sample_rate=SR)
    filters = fb.make_filters()[:16, 0, :].detach().numpy()
    freqs = np.fft.rfftfreq(1024, 1 / SR)

    peaks = []
    for kernel in filters:
        response = np.abs(np.fft.rfft(kernel, n=1024))
        peaks.append(freqs[int(response.argmax())])
    # Mel-spaced initialisation means peaks increase with filter index.
    assert peaks == sorted(peaks)
    assert peaks[0] < peaks[-1]


def test_sinc_filterbank_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ParamSincFB(n_filters=31, kernel_size=101)
    with pytest.raises(ValueError):
        ParamSincFB(n_filters=32, kernel_size=100)


# ---------------------------------------------------------------- components
def test_preemphasis_is_a_highpass():
    t = np.arange(SR) / SR
    low = torch.from_numpy((np.sin(2 * np.pi * 50 * t)).astype(np.float32)).unsqueeze(0)
    high = torch.from_numpy((np.sin(2 * np.pi * 6000 * t)).astype(np.float32)).unsqueeze(0)
    pre = PreEmphasis()
    assert float(pre(high).abs().mean()) > float(pre(low).abs().mean())


def test_afms_preserves_shape():
    x = torch.randn(2, 16, 50)
    assert AFMS(16)(x).shape == x.shape


@pytest.mark.parametrize("pool", [None, 3])
def test_bottle2neck_shapes(pool):
    block = Bottle2neck(16, 32, kernel_size=3, dilation=2, scale=4, pool=pool)
    out = block(torch.randn(2, 16, 90))
    assert out.shape[:2] == (2, 32)
    assert out.shape[2] == (90 // pool if pool else 90)


def test_backbone_embedding_dimension():
    model = RawNet3(out_dim=48, channels=64, model_scale=4, sinc_filters=32, sinc_kernel=101)
    assert model(torch.randn(2, SR) * 0.1).shape == (2, 48)
