"""Tests for audio RandAugment (Section 4.2, Fig. 10) and the corruption battery."""

import numpy as np
import pytest

from fsat.randaugment import (
    CORRUPTION_SUITE,
    RANDAUGMENT_OPS,
    TRANSFORMS,
    apply_transform,
    pitch_shift,
    rand_augment,
    time_stretch,
)

SR = 16000


@pytest.fixture
def tone():
    t = np.arange(SR * 2) / SR
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class _FixedRng:
    """Deterministic stand-in that returns a chosen value from uniform()."""

    def __init__(self, value):
        self.value = value

    def uniform(self, low, high):
        return self.value


def test_randaugment_pool_matches_paper_listing():
    """Fig. 10 lists exactly 22 transforms."""
    assert len(RANDAUGMENT_OPS) == 22
    assert len(set(RANDAUGMENT_OPS)) == 22
    assert set(RANDAUGMENT_OPS) <= set(TRANSFORMS)


def test_corruption_suite_has_24_entries():
    """Fig. 7a charts 24 corruption types."""
    assert len(CORRUPTION_SUITE) == 24
    assert len(set(CORRUPTION_SUITE)) == 24
    assert set(CORRUPTION_SUITE) <= set(TRANSFORMS)


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_every_transform_is_length_preserving_and_finite(tone, name):
    out = apply_transform(name, tone, SR, np.random.default_rng(0))
    assert out.shape == tone.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_every_transform_actually_changes_the_signal(tone, name):
    """A no-op transform would silently weaken augmentation."""
    out = apply_transform(name, tone, SR, np.random.default_rng(0))
    assert not np.allclose(out, tone, atol=1e-6), f"{name} was a no-op"


def test_unknown_transform_raises(tone):
    with pytest.raises(KeyError):
        apply_transform("does_not_exist", tone, SR, np.random.default_rng(0))


def test_probability_zero_is_identity(tone):
    out = rand_augment(tone, SR, n=3, p=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(out, tone)


def test_probability_one_always_augments(tone):
    out = rand_augment(tone, SR, n=2, p=1.0, rng=np.random.default_rng(0))
    assert not np.allclose(out, tone)


def test_zero_transforms_is_identity(tone):
    assert np.array_equal(rand_augment(tone, SR, n=0, p=1.0, rng=np.random.default_rng(0)), tone)


def test_randaugment_is_reproducible(tone):
    a = rand_augment(tone, SR, n=3, p=1.0, rng=np.random.default_rng(7))
    b = rand_augment(tone, SR, n=3, p=1.0, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_randaugment_preserves_length(tone):
    for n in (1, 2, 5):
        assert rand_augment(tone, SR, n=n, p=1.0, rng=np.random.default_rng(n)).shape == tone.shape


def _peak_hz(signal):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return np.fft.rfftfreq(signal.size, 1 / SR)[int(spectrum.argmax())]


@pytest.mark.parametrize("semitones", [-4, -2, 2, 4])
def test_pitch_shift_moves_pitch_in_the_right_direction(tone, semitones):
    out = pitch_shift(tone, SR, _FixedRng(semitones))
    expected = 440.0 * 2 ** (semitones / 12.0)
    assert out.size == tone.size
    assert abs(_peak_hz(out) - expected) < 15.0


@pytest.mark.parametrize("rate", [0.8, 1.25])
def test_time_stretch_preserves_pitch(tone, rate):
    out = time_stretch(tone, SR, _FixedRng(rate))
    assert out.size == tone.size  # refitted to the original length
    assert abs(_peak_hz(out) - 440.0) < 15.0


def test_polarity_inversion_is_exact(tone):
    out = apply_transform("polarity_inversion", tone, SR, np.random.default_rng(0))
    assert np.allclose(out, -tone)


def test_bit_crush_quantizes(tone):
    out = apply_transform("bit_crush", tone, SR, np.random.default_rng(0))
    # Quantization collapses the signal onto far fewer distinct values.
    assert np.unique(out).size < np.unique(tone).size


def test_time_mask_zeroes_a_contiguous_span(tone):
    out = apply_transform("time_mask", tone, SR, np.random.default_rng(0))
    assert (out == 0).sum() > (tone == 0).sum()


def test_aliasing_folds_energy_downward():
    """Downsampling without an anti-alias filter must move a high tone lower."""
    t = np.arange(SR) / SR
    high = (0.5 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    out = apply_transform("aliasing", high, SR, np.random.default_rng(0))
    assert _peak_hz(out) < 7000.0
