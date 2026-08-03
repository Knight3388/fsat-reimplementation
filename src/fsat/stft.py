"""Band-selective STFT injection operator.

Implements the ``s(x, delta, f_l, f_u)`` operator of Section 4.1 of
"I Can Hear You: Selective Robust Training for Deepfake Audio Detection"
(Zhang et al., arXiv:2411.00121v1).

The paper defines, for an input waveform ``x`` with STFT ``X``, magnitude
``X_rho`` and phase ``X_phi``:

    r_l = floor(f_l * n_fft / sr)          r_u = ceil(f_u * n_fft / sr)

    D_kk = 1 if r_l <= k <= r_u else 0

    delta_s = M(delta, r_l, r_u) = D . delta

    X'      = (X_rho + delta_s) . exp(j * X_phi)

    x'      = ISTFT(X')

Only the *magnitude* is perturbed, the phase is carried through untouched.
The paper's ablation (Fig. 8a) reports that magnitude attacks dominate
phase attacks, which is why F-SAT targets magnitude.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class BandSelectiveSTFT(nn.Module):
    """Differentiable STFT/ISTFT pair with a frequency-band selection mask.

    Analysis is used to obtain ``(magnitude, phasor)`` of the clean signal.
    Synthesis re-assembles a waveform from a *perturbed* magnitude and the
    original phasor, and is differentiable with respect to that magnitude,
    which is what lets PGD backpropagate into ``delta``.

    Phase is represented as a unit complex "phasor" ``X / |X|`` rather than an
    angle. This is mathematically identical to ``exp(j * X_phi)`` but avoids
    the ``atan2`` gradient singularity at ``|X| = 0``.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int | None = None,
        win_length: int | None = None,
        sample_rate: int = 16000,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.n_fft // 4
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.sample_rate = int(sample_rate)
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    # ------------------------------------------------------------------ #
    # Frequency-bin bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def num_bins(self) -> int:
        """Number of non-negative-frequency bins, ``n_fft // 2 + 1``."""
        return self.n_fft // 2 + 1

    def band_indices(self, f_lo: float, f_hi: float) -> Tuple[int, int]:
        """Return ``(r_l, r_u)``, the inclusive bin bounds for ``[f_lo, f_hi]`` Hz.

        Follows the paper's formulas verbatim, then clamps to the valid
        one-sided bin range so that a request such as ``(6000, 8000)`` at
        ``sr = 16000`` cannot run past Nyquist.
        """
        if f_hi < f_lo:
            raise ValueError(f"f_hi ({f_hi}) must be >= f_lo ({f_lo})")
        r_l = math.floor(f_lo * self.n_fft / self.sample_rate)
        r_u = math.ceil(f_hi * self.n_fft / self.sample_rate)
        r_l = max(0, min(r_l, self.num_bins - 1))
        r_u = max(0, min(r_u, self.num_bins - 1))
        return r_l, r_u

    def band_mask(
        self,
        f_lo: float,
        f_hi: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Diagonal selection matrix ``D`` of Eq. (3), returned as a ``(F, 1)`` vector.

        ``D`` is diagonal, so storing the diagonal and multiplying elementwise is
        equivalent to the matrix product ``D . delta`` and far cheaper.
        The trailing singleton axis broadcasts over STFT frames.
        """
        r_l, r_u = self.band_indices(f_lo, f_hi)
        mask = torch.zeros(self.num_bins, 1, device=device, dtype=dtype or torch.float32)
        mask[r_l : r_u + 1] = 1.0
        return mask

    # ------------------------------------------------------------------ #
    # Analysis / synthesis
    # ------------------------------------------------------------------ #
    def analyze(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Waveform ``(B, T)`` -> ``(magnitude, phasor)``, each ``(B, F, N)``."""
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"expected (B, T) or (B, 1, T) waveform, got {tuple(x.shape)}")
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device, x.dtype),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = spec.abs()
        phasor = spec / (magnitude + self.eps)
        return magnitude, phasor

    def synthesize(self, magnitude: torch.Tensor, phasor: torch.Tensor, length: int) -> torch.Tensor:
        """``(magnitude, phasor)`` -> waveform ``(B, length)``, differentiable in ``magnitude``."""
        spec = magnitude.to(phasor.dtype) * phasor
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(magnitude.device, magnitude.dtype),
            center=True,
            normalized=False,
            onesided=True,
            length=length,
        )

    # ------------------------------------------------------------------ #
    # The s(x, delta, f_l, f_u) operator
    # ------------------------------------------------------------------ #
    def inject(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        f_lo: float,
        f_hi: float,
        clamp_magnitude: bool = True,
    ) -> torch.Tensor:
        """Frequency-selective injection ``x' = s(x, delta, f_l, f_u)``.

        Args:
            x: clean waveform ``(B, T)``.
            delta: magnitude-domain perturbation ``(B, F, N)``.
            f_lo, f_hi: band edges in Hz.
            clamp_magnitude: floor the perturbed magnitude at zero. The paper
                does not specify this. Without it, a large negative ``delta``
                drives ``X_rho + delta_s`` below zero, which is not a magnitude
                but a 180-degree phase flip smuggled in through the magnitude
                channel. Clamping keeps the perturbation inside the magnitude
                subspace the method claims to operate on. Set ``False`` to
                reproduce the unclamped reading of Eq. (5).

        Returns:
            Perturbed waveform ``(B, T)``.
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        magnitude, phasor = self.analyze(x)
        mask = self.band_mask(f_lo, f_hi, device=x.device, dtype=magnitude.dtype)
        perturbed = magnitude + mask * delta
        if clamp_magnitude:
            perturbed = perturbed.clamp_min(0.0)
        return self.synthesize(perturbed, phasor, length=x.size(-1))
