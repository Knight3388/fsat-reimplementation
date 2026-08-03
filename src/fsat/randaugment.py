"""RandAugment for audio, plus the corruption battery used for robust evaluation.

Section 4.2 of arXiv:2411.00121v1 adapts image RandAugment (Cubuk et al., 2020)
to audio. The paper's Fig. 10 gives the reference implementation:

    audio_transforms = [
        'background_noise', 'color_noise', 'short_noise', 'gaussian_noise',
        'air_absorption', 'room_simulator', 'band_pass', 'band_stop',
        'high_pass', 'low_pass', 'high_shelf', 'low_shelf', 'peaking',
        'aliasing', 'bit_crush', 'clip', 'tanh_distortion', 'gain_transition',
        'seven_band_parametric_EQ', 'pitch_shift', 'time_mask', 'time_stretch'
    ]

    def rand_augment_audio(sample, N, p):
        operations = np.random.choice(audio_transforms, N)
        return operations(sample) if np.random.random() < p else sample

Two faithfulness notes on that snippet:

1. ``np.random.choice`` without ``replace=False`` samples *with* replacement, so
   the same transform can be drawn twice. :func:`rand_augment` reproduces this
   by default (``replace=True``).
2. The single ``np.random.random() < p`` gate is evaluated once for the whole
   chain, not per transform, so ``p`` is the probability that *any* augmentation
   happens. We keep that as the default (``per_op_probability=False``) and offer
   a per-transform gate as an option.

Every transform preserves input length so augmented clips stay batchable.
``time_stretch`` therefore stretches and then crops or pads back to the
original duration.

The corruption battery in :data:`CORRUPTION_SUITE` is the 24-corruption
evaluation set of Fig. 7a, which overlaps the RandAugment pool but adds
``gain``, ``polarity_inversion``, ``limiter`` and ``clipping_distortion``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import signal

Waveform = np.ndarray
Transform = Callable[[Waveform, int, np.random.Generator], Waveform]

_EPS = 1e-12


# ====================================================================== #
# Low-level helpers
# ====================================================================== #
def _as_float32(x: Waveform) -> Waveform:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _fit_length(x: Waveform, length: int) -> Waveform:
    """Crop or zero-pad ``x`` to exactly ``length`` samples."""
    if x.size == length:
        return x
    if x.size > length:
        return x[:length]
    return np.pad(x, (0, length - x.size))


def _biquad(x: Waveform, b: Sequence[float], a: Sequence[float]) -> Waveform:
    return signal.lfilter(np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64), x).astype(np.float32)


def _rbj_coeffs(kind: str, f0: float, sr: int, gain_db: float, q: float):
    """Robert Bristow-Johnson audio EQ cookbook biquad coefficients."""
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_a = np.sqrt(a_)

    if kind == "peaking":
        b = [1 + alpha * a_, -2 * cos_w0, 1 - alpha * a_]
        a = [1 + alpha / a_, -2 * cos_w0, 1 - alpha / a_]
    elif kind == "low_shelf":
        b = [
            a_ * ((a_ + 1) - (a_ - 1) * cos_w0 + 2 * sqrt_a * alpha),
            2 * a_ * ((a_ - 1) - (a_ + 1) * cos_w0),
            a_ * ((a_ + 1) - (a_ - 1) * cos_w0 - 2 * sqrt_a * alpha),
        ]
        a = [
            (a_ + 1) + (a_ - 1) * cos_w0 + 2 * sqrt_a * alpha,
            -2 * ((a_ - 1) + (a_ + 1) * cos_w0),
            (a_ + 1) + (a_ - 1) * cos_w0 - 2 * sqrt_a * alpha,
        ]
    elif kind == "high_shelf":
        b = [
            a_ * ((a_ + 1) + (a_ - 1) * cos_w0 + 2 * sqrt_a * alpha),
            -2 * a_ * ((a_ - 1) + (a_ + 1) * cos_w0),
            a_ * ((a_ + 1) + (a_ - 1) * cos_w0 - 2 * sqrt_a * alpha),
        ]
        a = [
            (a_ + 1) - (a_ - 1) * cos_w0 + 2 * sqrt_a * alpha,
            2 * ((a_ - 1) - (a_ + 1) * cos_w0),
            (a_ + 1) - (a_ - 1) * cos_w0 - 2 * sqrt_a * alpha,
        ]
    else:
        raise ValueError(f"unknown biquad kind: {kind}")
    return b, a


def _butter(x: Waveform, cutoff, sr: int, btype: str, order: int = 4) -> Waveform:
    nyq = sr / 2.0
    if isinstance(cutoff, (tuple, list, np.ndarray)):
        wn = [float(np.clip(c / nyq, 1e-4, 0.9999)) for c in cutoff]
        if wn[0] >= wn[1]:
            return x
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.9999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _resample(x: Waveform, ratio: float) -> Waveform:
    """Resample by ``ratio`` (>1 lengthens) via polyphase filtering."""
    if abs(ratio - 1.0) < 1e-6:
        return x
    frac = Fraction(float(ratio)).limit_denominator(200)
    up, down = max(1, frac.numerator), max(1, frac.denominator)
    return signal.resample_poly(x, up, down).astype(np.float32)


def _stft_np(x: Waveform, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft + 1)[:-1].astype(np.float64)
    xp = np.pad(x.astype(np.float64), n_fft // 2, mode="reflect")
    n_frames = 1 + max(0, (len(xp) - n_fft) // hop)
    frames = np.stack([xp[i * hop : i * hop + n_fft] * win for i in range(n_frames)], axis=1)
    return np.fft.rfft(frames, n=n_fft, axis=0)


def _istft_np(spec: np.ndarray, n_fft: int, hop: int, length: Optional[int] = None) -> Waveform:
    win = np.hanning(n_fft + 1)[:-1].astype(np.float64)
    frames = np.fft.irfft(spec, n=n_fft, axis=0)
    n_frames = frames.shape[1]
    out = np.zeros(n_fft + hop * (n_frames - 1))
    norm = np.zeros_like(out)
    for i in range(n_frames):
        out[i * hop : i * hop + n_fft] += frames[:, i] * win
        norm[i * hop : i * hop + n_fft] += win**2
    out = out / np.maximum(norm, 1e-8)
    out = out[n_fft // 2 : len(out) - n_fft // 2]
    if length is not None:
        out = _fit_length(out.astype(np.float32), length)
    return out.astype(np.float32)


def _phase_vocoder(x: Waveform, rate: float, n_fft: int = 1024, hop: int = 256) -> Waveform:
    """Time-scale ``x`` by ``rate`` (>1 = faster/shorter) preserving pitch."""
    if abs(rate - 1.0) < 1e-6 or x.size < n_fft * 2:
        return x
    spec = _stft_np(x, n_fft, hop)
    n_bins, n_frames = spec.shape
    steps = np.arange(0, n_frames - 1, rate)
    magnitude, angle = np.abs(spec), np.angle(spec)
    expected = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft

    out = np.zeros((n_bins, len(steps)), dtype=np.complex128)
    accumulator = angle[:, 0].copy()
    for i, t in enumerate(steps):
        i0 = int(np.floor(t))
        frac = t - i0
        mag = (1.0 - frac) * magnitude[:, i0] + frac * magnitude[:, i0 + 1]
        out[:, i] = mag * np.exp(1j * accumulator)
        dphi = angle[:, i0 + 1] - angle[:, i0] - expected
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        accumulator += expected + dphi
    return _istft_np(out, n_fft, hop)


def _rms(x: Waveform) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + _EPS))


def _mix_at_snr(x: Waveform, noise: Waveform, snr_db: float) -> Waveform:
    noise = _fit_length(noise, x.size)
    scale = _rms(x) / (_rms(noise) + _EPS) * (10.0 ** (-snr_db / 20.0))
    return (x + scale * noise).astype(np.float32)


def _colored_noise(n: int, exponent: float, rng: np.random.Generator) -> Waveform:
    """Noise with a ``1 / f**exponent`` power spectrum (0 = white, 1 = pink, 2 = brown)."""
    freqs = np.fft.rfftfreq(n)
    scaling = np.ones_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = freqs[nonzero] ** (-exponent / 2.0)
    scaling[0] = 0.0
    spectrum = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) * scaling
    return np.fft.irfft(spectrum, n=n).astype(np.float32)


# ====================================================================== #
# The 22 RandAugment transforms
# ====================================================================== #
def background_noise(
    x: Waveform, sr: int, rng: np.random.Generator, noise_pool: Optional[Sequence[Waveform]] = None
) -> Waveform:
    """Mix in a background recording at 5-30 dB SNR.

    With no corpus supplied (the smoke-test path) this falls back to pink-ish
    noise, which keeps the transform functional without a data dependency.
    Pass ``noise_pool`` to use real recordings, as the paper does.
    """
    snr = rng.uniform(5.0, 30.0)
    if noise_pool:
        noise = _as_float32(noise_pool[rng.integers(len(noise_pool))])
        if noise.size < x.size:
            noise = np.tile(noise, int(np.ceil(x.size / max(noise.size, 1))))
        offset = int(rng.integers(0, max(1, noise.size - x.size + 1)))
        noise = noise[offset : offset + x.size]
    else:
        noise = _colored_noise(x.size, exponent=rng.uniform(0.5, 1.5), rng=rng)
    return _mix_at_snr(x, noise, snr)


def color_noise(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Add noise with a random spectral slope (white through brown)."""
    return _mix_at_snr(x, _colored_noise(x.size, rng.uniform(-2.0, 2.0), rng), rng.uniform(3.0, 30.0))


def short_noise(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Add one or more brief noise bursts rather than continuous noise."""
    out = x.copy()
    for _ in range(int(rng.integers(1, 4))):
        burst_len = int(rng.uniform(0.02, 0.25) * sr)
        burst_len = min(burst_len, x.size)
        if burst_len < 2:
            continue
        start = int(rng.integers(0, x.size - burst_len + 1))
        segment = out[start : start + burst_len]
        burst = rng.normal(size=burst_len).astype(np.float32)
        envelope = np.hanning(burst_len).astype(np.float32)
        out[start : start + burst_len] = _mix_at_snr(segment, burst * envelope, rng.uniform(0.0, 20.0))
    return out


def gaussian_noise(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Additive white Gaussian noise at 5-40 dB SNR."""
    return _mix_at_snr(x, rng.normal(size=x.size).astype(np.float32), rng.uniform(5.0, 40.0))


def air_absorption(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Distance-dependent high-frequency roll-off (Appendix A.1).

    Air attenuates high frequencies more than low ones, so this is modelled as a
    first-order low-pass whose cutoff falls as the simulated distance grows.
    """
    distance_m = rng.uniform(1.0, 20.0)
    cutoff = float(np.clip(sr / 2.0 * np.exp(-distance_m / 12.0), 1000.0, sr / 2.0 - 1.0))
    return _butter(x, cutoff, sr, "lowpass", order=1)


def room_simulator(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Convolve with a synthetic room impulse response (Appendix A.1).

    The RIR is exponentially decaying noise with a direct-path impulse, which
    reproduces the reverberation tail without needing a measured RIR corpus.
    """
    rt60 = rng.uniform(0.1, 0.8)
    n = int(rt60 * sr)
    if n < 8:
        return x
    t = np.arange(n) / sr
    rir = rng.normal(size=n).astype(np.float32) * np.exp(-6.9078 * t / rt60).astype(np.float32)
    rir[0] += 1.0
    rir /= np.linalg.norm(rir) + _EPS
    wet = signal.fftconvolve(x, rir, mode="full")[: x.size].astype(np.float32)
    # Preserve loudness so reverb does not double as a gain change.
    return (wet * (_rms(x) / (_rms(wet) + _EPS))).astype(np.float32)


def band_pass(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    low = rng.uniform(100.0, 1500.0)
    high = rng.uniform(low * 2.0, sr / 2.0 * 0.95)
    return _butter(x, (low, high), sr, "bandpass")


def band_stop(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    low = rng.uniform(200.0, 3000.0)
    high = rng.uniform(low * 1.5, min(low * 4.0, sr / 2.0 * 0.95))
    return _butter(x, (low, high), sr, "bandstop")


def high_pass(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    return _butter(x, rng.uniform(50.0, 2000.0), sr, "highpass")


def low_pass(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    return _butter(x, rng.uniform(1500.0, sr / 2.0 * 0.95), sr, "lowpass")


def high_shelf(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    b, a = _rbj_coeffs("high_shelf", rng.uniform(2000.0, sr / 2.0 * 0.9), sr, rng.uniform(-18.0, 18.0), 0.707)
    return _biquad(x, b, a)


def low_shelf(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    b, a = _rbj_coeffs("low_shelf", rng.uniform(50.0, 1000.0), sr, rng.uniform(-18.0, 18.0), 0.707)
    return _biquad(x, b, a)


def peaking(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Boost or cut a band around a centre frequency (Appendix A.1, "Peaking")."""
    b, a = _rbj_coeffs(
        "peaking", rng.uniform(100.0, sr / 2.0 * 0.9), sr, rng.uniform(-18.0, 18.0), rng.uniform(0.5, 4.0)
    )
    return _biquad(x, b, a)


def aliasing(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Downsample without an anti-alias filter, then restore the rate (Appendix A.1).

    Violating ``f_sample > 2 * f_max`` folds high frequencies down into the
    passband. Fig. 7a reports this as the single most damaging corruption,
    because it destroys exactly the high-frequency evidence detectors rely on.
    """
    factor = int(rng.integers(2, 5))
    decimated = x[::factor]
    if decimated.size < 2:
        return x
    idx = np.linspace(0, decimated.size - 1, num=x.size)
    return np.interp(idx, np.arange(decimated.size), decimated).astype(np.float32)


def bit_crush(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Reduce bit depth, introducing quantization error (Appendix A.1)."""
    bits = int(rng.integers(4, 13))
    levels = 2 ** (bits - 1)
    return (np.round(np.clip(x, -1.0, 1.0) * levels) / levels).astype(np.float32)


def clip(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Hard-clip at a percentile of the amplitude distribution."""
    threshold = float(np.percentile(np.abs(x), rng.uniform(90.0, 99.9))) + _EPS
    return np.clip(x, -threshold, threshold).astype(np.float32)


def tanh_distortion(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Soft clipping ``y = tanh(k * x)`` (Appendix A.1)."""
    k = rng.uniform(1.0, 20.0)
    out = np.tanh(k * x).astype(np.float32)
    return (out * (_rms(x) / (_rms(out) + _EPS))).astype(np.float32)


def gain_transition(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Smoothly ramp the level over part of the clip (Appendix A.1)."""
    start = int(rng.integers(0, max(1, x.size - 1)))
    end = int(rng.integers(start + 1, x.size + 1))
    gain_db = rng.uniform(-24.0, 12.0)
    envelope = np.ones(x.size, dtype=np.float32)
    envelope[start:end] = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
    envelope[end:] = 1.0
    envelope = 10.0 ** (gain_db * envelope / 20.0)
    return (x * envelope).astype(np.float32)


def seven_band_parametric_eq(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Cascade of seven peaking filters, ``H(f) = prod_i H_i(f)`` (Appendix A.1)."""
    centers = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    out = x
    for f0 in centers:
        if f0 >= sr / 2.0 * 0.95:
            continue
        b, a = _rbj_coeffs("peaking", f0, sr, rng.uniform(-12.0, 12.0), 1.0)
        out = _biquad(out, b, a)
    return out


def pitch_shift(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Shift pitch by +/- 4 semitones at constant duration.

    Implemented as a phase-vocoder time stretch followed by resampling, the
    standard construction. To raise pitch by a ratio ``r``: lengthen by ``r``
    at constant pitch, then resample back down by ``r``, which compresses the
    time axis and scales the spectrum up by ``r`` at the original duration.
    """
    semitones = rng.uniform(-4.0, 4.0)
    if abs(semitones) < 1e-3:
        return x
    ratio = 2.0 ** (semitones / 12.0)  # desired pitch ratio, >1 raises pitch
    stretched = _phase_vocoder(x, 1.0 / ratio)  # rate < 1 lengthens
    shifted = _resample(stretched, 1.0 / ratio)  # ratio < 1 shortens
    return _fit_length(shifted, x.size)


def time_mask(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Silence a contiguous segment (Appendix A.1, "Time Mask")."""
    out = x.copy()
    span = int(rng.uniform(0.01, 0.15) * x.size)
    if span < 1:
        return out
    start = int(rng.integers(0, x.size - span + 1))
    out[start : start + span] = 0.0
    return out


def time_stretch(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Change tempo at constant pitch, then refit to the original length."""
    rate = rng.uniform(0.8, 1.25)
    return _fit_length(_phase_vocoder(x, rate), x.size)


# ====================================================================== #
# Extra corruptions used by the Fig. 7a evaluation battery
# ====================================================================== #
def gain(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    return (x * 10.0 ** (rng.uniform(-18.0, 6.0) / 20.0)).astype(np.float32)


def polarity_inversion(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    return (-x).astype(np.float32)


def limiter(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Soft-knee limiting above a threshold."""
    threshold = float(np.percentile(np.abs(x), rng.uniform(70.0, 95.0))) + _EPS
    over = np.abs(x) > threshold
    out = x.copy()
    out[over] = np.sign(x[over]) * (threshold + np.tanh((np.abs(x[over]) - threshold) / threshold) * threshold * 0.3)
    return out.astype(np.float32)


def clipping_distortion(x: Waveform, sr: int, rng: np.random.Generator) -> Waveform:
    """Clip a fixed *percentile* of samples, the audiomentations formulation."""
    percentile = rng.uniform(0.0, 40.0)
    threshold = float(np.percentile(np.abs(x), 100.0 - percentile)) + _EPS
    return np.clip(x, -threshold, threshold).astype(np.float32)


# ====================================================================== #
# Registries
# ====================================================================== #
TRANSFORMS: Dict[str, Transform] = {
    "background_noise": background_noise,
    "color_noise": color_noise,
    "short_noise": short_noise,
    "gaussian_noise": gaussian_noise,
    "air_absorption": air_absorption,
    "room_simulator": room_simulator,
    "band_pass": band_pass,
    "band_stop": band_stop,
    "high_pass": high_pass,
    "low_pass": low_pass,
    "high_shelf": high_shelf,
    "low_shelf": low_shelf,
    "peaking": peaking,
    "aliasing": aliasing,
    "bit_crush": bit_crush,
    "clip": clip,
    "tanh_distortion": tanh_distortion,
    "gain_transition": gain_transition,
    "seven_band_parametric_EQ": seven_band_parametric_eq,
    "pitch_shift": pitch_shift,
    "time_mask": time_mask,
    "time_stretch": time_stretch,
    # Fig. 7a corruption battery only, not part of the RandAugment pool.
    "gain": gain,
    "polarity_inversion": polarity_inversion,
    "limiter": limiter,
    "clipping_distortion": clipping_distortion,
}

#: The exact 22-transform pool of Fig. 10, in the paper's order.
RANDAUGMENT_OPS: List[str] = [
    "background_noise",
    "color_noise",
    "short_noise",
    "gaussian_noise",
    "air_absorption",
    "room_simulator",
    "band_pass",
    "band_stop",
    "high_pass",
    "low_pass",
    "high_shelf",
    "low_shelf",
    "peaking",
    "aliasing",
    "bit_crush",
    "clip",
    "tanh_distortion",
    "gain_transition",
    "seven_band_parametric_EQ",
    "pitch_shift",
    "time_mask",
    "time_stretch",
]

#: The 24 corruptions charted in Fig. 7a, used for robust evaluation.
CORRUPTION_SUITE: List[str] = [
    "time_stretch",
    "aliasing",
    "time_mask",
    "background_noise",
    "color_noise",
    "gaussian_noise",
    "short_noise",
    "air_absorption",
    "band_pass",
    "band_stop",
    "bit_crush",
    "clipping_distortion",
    "clip",
    "gain",
    "gain_transition",
    "high_pass",
    "high_shelf",
    "low_pass",
    "low_shelf",
    "peaking",
    "polarity_inversion",
    "room_simulator",
    "seven_band_parametric_EQ",
    "tanh_distortion",
]


def apply_transform(
    name: str, x: Waveform, sr: int, rng: np.random.Generator, **kwargs
) -> Waveform:
    """Apply a single named transform, guaranteeing the output length matches."""
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; available: {sorted(TRANSFORMS)}")
    out = TRANSFORMS[name](_as_float32(x), sr, rng, **kwargs)
    out = _fit_length(_as_float32(out), int(np.asarray(x).size))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rand_augment(
    sample: Waveform,
    sr: int = 16000,
    n: int = 2,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    pool: Optional[Iterable[str]] = None,
    replace: bool = True,
    per_op_probability: bool = False,
) -> Waveform:
    """RandAugment for audio (Section 4.2, Fig. 10).

    Args:
        sample: 1-D waveform.
        sr: sample rate in Hz.
        n: number of transformations to draw, the paper's ``N``.
        p: probability of applying augmentation, the paper's ``p``. By default
            this gates the whole chain once, matching the reference snippet.
        rng: numpy generator, for reproducibility.
        pool: transform names to draw from. Defaults to :data:`RANDAUGMENT_OPS`.
        replace: sample with replacement, as ``np.random.choice`` does by default.
        per_op_probability: gate each transform independently instead of gating
            the chain once. Not the paper's behaviour, offered as an option.

    Returns:
        Augmented waveform, same length and dtype ``float32``.
    """
    rng = rng or np.random.default_rng()
    x = _as_float32(sample)
    names = list(pool) if pool is not None else RANDAUGMENT_OPS
    if n <= 0 or not names:
        return x

    if not per_op_probability and rng.random() >= p:
        return x

    chosen = rng.choice(names, size=min(n, len(names)) if not replace else n, replace=replace)
    for name in chosen:
        if per_op_probability and rng.random() >= p:
            continue
        x = apply_transform(str(name), x, sr, rng)
    return x


__all__ = [
    "TRANSFORMS",
    "RANDAUGMENT_OPS",
    "CORRUPTION_SUITE",
    "apply_transform",
    "rand_augment",
]
