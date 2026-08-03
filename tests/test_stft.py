"""Tests for the band-selective STFT operator, Eq. (1)-(5) of the paper."""

import math

import numpy as np
import pytest
import torch

from fsat.stft import BandSelectiveSTFT

SR = 16000


@pytest.fixture
def stft():
    return BandSelectiveSTFT(n_fft=1024, hop_length=256, sample_rate=SR)


def test_roundtrip_is_lossless(stft):
    x = torch.randn(3, SR) * 0.1
    mag, phasor = stft.analyze(x)
    assert torch.allclose(stft.synthesize(mag, phasor, x.size(-1)), x, atol=1e-5)


def test_band_indices_match_paper_formula(stft):
    """r_l = floor(f_l * n_fft / sr), r_u = ceil(f_u * n_fft / sr)."""
    for f_lo, f_hi in [(0, 8000), (2000, 8000), (4000, 8000), (6000, 8000), (300, 3400)]:
        r_l, r_u = stft.band_indices(f_lo, f_hi)
        expected_l = math.floor(f_lo * stft.n_fft / SR)
        expected_u = min(math.ceil(f_hi * stft.n_fft / SR), stft.num_bins - 1)
        assert (r_l, r_u) == (expected_l, expected_u)


def test_band_indices_clamped_to_nyquist(stft):
    r_l, r_u = stft.band_indices(0, 999_999)
    assert r_u == stft.num_bins - 1


def test_band_indices_reject_inverted_range(stft):
    with pytest.raises(ValueError):
        stft.band_indices(8000, 4000)


def test_band_mask_is_binary_and_contiguous(stft):
    mask = stft.band_mask(4000, 8000)
    assert mask.shape == (stft.num_bins, 1)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    r_l, r_u = stft.band_indices(4000, 8000)
    assert mask[r_l : r_u + 1].all() and mask[:r_l].sum() == 0


def test_zero_perturbation_is_identity(stft):
    x = torch.randn(2, SR) * 0.1
    mag, _ = stft.analyze(x)
    assert torch.allclose(stft.inject(x, torch.zeros_like(mag), 4000, 8000), x, atol=1e-5)


def test_gradient_flows_only_inside_the_band(stft):
    x = torch.randn(2, SR) * 0.1
    mag, _ = stft.analyze(x)
    delta = torch.zeros_like(mag, requires_grad=True)
    stft.inject(x, delta, 4000, 8000).pow(2).sum().backward()

    r_l, r_u = stft.band_indices(4000, 8000)
    assert delta.grad[:, r_l : r_u + 1].abs().sum() > 0
    assert delta.grad[:, :r_l].abs().max() == 0


def test_perturbation_energy_stays_in_band(stft):
    """A band-limited magnitude delta must not leak energy below f_lo."""
    x = torch.randn(2, SR) * 0.1
    mag, _ = stft.analyze(x)
    delta = torch.full_like(mag, 0.05)

    residual = stft.inject(x, delta, 4000, 8000) - x
    spec = torch.stft(
        residual, 1024, 256, window=torch.hann_window(1024), return_complex=True
    ).abs().mean(dim=(0, 2))
    freqs = torch.from_numpy(np.fft.rfftfreq(1024, 1 / SR))

    below = spec[freqs < 4000].pow(2).sum()
    total = spec.pow(2).sum()
    # Some spill is unavoidable from STFT window sidelobes and overlap-add.
    assert float(below / total) < 0.02


def test_magnitude_clamping_prevents_negative_magnitude(stft):
    x = torch.randn(2, SR) * 0.1
    mag, phasor = stft.analyze(x)
    huge = torch.full_like(mag, -100.0)

    clamped = stft.inject(x, huge, 0, 8000, clamp_magnitude=True)
    # A fully suppressed magnitude means (near-)silence.
    assert float(clamped.abs().max()) < 1e-3

    unclamped = stft.inject(x, huge, 0, 8000, clamp_magnitude=False)
    assert float(unclamped.abs().max()) > 1.0


def test_accepts_channel_dimension(stft):
    x = torch.randn(2, 1, SR) * 0.1
    mag, _ = stft.analyze(x)
    assert stft.inject(x, torch.zeros_like(mag), 4000, 8000).shape == (2, SR)
