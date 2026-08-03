"""Adversarial attacks on waveform deepfake detectors.

Implements the three perturbation domains compared in Section 4.1 and
Fig. 8a of arXiv:2411.00121v1:

* :class:`FrequencySelectiveAttack` -- PGD on the STFT *magnitude*, confined
  to a frequency band ``[f_lo, f_hi]``. This is the attack F-SAT trains
  against, and with ``f_lo=0`` it degenerates to a full-band spectral attack.
* :class:`TimeDomainAttack` -- ordinary PGD on the raw waveform, ``x' = x + delta``.
* :class:`PhaseAttack` -- PGD on the STFT *phase*, kept as the ablation arm
  the paper reports as least effective.

All three share the PGD skeleton of Eq. (6):

    x'_{n+1} = Pi_{x+S}( x'_n + alpha * M( sgn( grad L(s(x, delta, f_l, f_u), y) ) ) )

Note on the update variable. Eq. (6) is written with ``x'`` on the left but
takes the mask ``M`` and the sign of the gradient on the right, while Fig. 4
shows the loop updating the *selective perturbation* ``delta`` that is added to
the magnitude. We follow Fig. 4 and iterate on ``delta`` in the masked
magnitude domain, which is the only reading under which the mask ``M`` and the
projection onto ``||x' - x|| <= epsilon`` are both well defined. The masked
magnitude update is exactly equivalent to Eq. (6) restricted to the band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft import BandSelectiveSTFT

# A detector maps a waveform (B, T) to class logits (B, C).
Detector = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AttackConfig:
    """PGD hyperparameters.

    Defaults follow the paper's tuned settings: Fig. 9a selects
    ``epsilon = 0.01`` as the attack magnitude giving the best
    accuracy/robustness balance, and Section 5.4 uses a handful of
    iterations. The appendix (Table 5) evaluates with
    ``epsilon = 1e-4, alpha = 4e-4`` in the frequency domain and
    ``epsilon = 1e-4, alpha = 4e-5`` in the time domain, over
    ``iters in {2, 5}`` and ``restarts in {1, 2}``.
    """

    epsilon: float = 0.01
    alpha: float = 0.004
    num_steps: int = 5
    num_restarts: int = 1
    random_start: bool = True
    clamp_magnitude: bool = True
    epsilon_mode: str = "absolute"
    """How to interpret ``epsilon`` for spectral attacks: ``"absolute"`` or ``"relative"``.

    The paper gives ``epsilon`` as a bare number on the STFT magnitude but never
    states the STFT normalisation or ``n_fft``. Unnormalised magnitudes scale
    with both, so the same absolute ``epsilon`` is a large perturbation on a
    quiet clip and a negligible one on a loud clip.

    ``"absolute"`` is the paper's literal reading and the default.
    ``"relative"`` rescales ``epsilon`` and ``alpha`` per utterance by the mean
    in-band magnitude, which makes attack strength invariant to input gain.
    Use it when a corpus has inconsistent levels.

    This does *not* make the attack invariant to ``n_fft`` (measured spread over
    ``n_fft`` in {512, 1024, 2048} is ~3.2x relative versus ~2.1x absolute), so
    ``n_fft`` must still be held fixed when comparing runs either way.

    Time-domain attacks ignore this setting: the waveform is already in [-1, 1].
    ``PhaseAttack`` also ignores it, since its epsilon is an angle in radians.
    """


class _PGDBase:
    """Shared multi-restart PGD driver.

    Subclasses supply the perturbation parameterisation by implementing
    :meth:`_init_delta` and :meth:`_render`.
    """

    def __init__(self, config: Optional[AttackConfig] = None) -> None:
        self.config = config or AttackConfig()

    # -- to be provided by subclasses ---------------------------------- #
    def _init_delta(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _render(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Map ``(clean waveform, perturbation)`` to the perturbed waveform."""
        raise NotImplementedError

    def _budget(self, x: torch.Tensor):
        """Return ``(epsilon, alpha)``, either scalars or per-sample tensors."""
        return self.config.epsilon, self.config.alpha

    @staticmethod
    def _project(delta: torch.Tensor, epsilon) -> torch.Tensor:
        """L-infinity projection onto the allowable set ``S``.

        ``epsilon`` may be a scalar or a tensor broadcastable to ``delta``,
        which is what allows a per-utterance budget.
        """
        if torch.is_tensor(epsilon):
            return torch.maximum(torch.minimum(delta, epsilon), -epsilon)
        return delta.clamp(-epsilon, epsilon)

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        model: Detector,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return the worst-case perturbed waveform, shape ``(B, T)``.

        The model is queried but never updated. Gradients w.r.t. model
        parameters are discarded, only the input-side gradient is used.
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        cfg = self.config
        was_training = isinstance(model, nn.Module) and model.training
        if isinstance(model, nn.Module):
            model.eval()

        epsilon, alpha = self._budget(x)

        # Per-sample running best (highest loss) across restarts.
        best_adv = x.detach().clone()
        best_loss = torch.full((x.size(0),), -float("inf"), device=x.device, dtype=x.dtype)

        for _ in range(max(1, cfg.num_restarts)):
            # _init_delta returns unit-scale noise (or zeros); the budget sets the scale.
            delta = self._project(self._init_delta(x) * epsilon, epsilon).detach()

            for _ in range(max(1, cfg.num_steps)):
                delta.requires_grad_(True)
                x_adv = self._render(x, delta)
                logits = model(x_adv)
                loss = (
                    loss_fn(logits, y)
                    if loss_fn is not None
                    else F.cross_entropy(logits, y, reduction="mean")
                )
                (grad,) = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)
                delta = self._project(delta.detach() + alpha * grad.sign(), epsilon).detach()

            with torch.no_grad():
                x_adv = self._render(x, delta)
                per_sample = F.cross_entropy(model(x_adv), y, reduction="none")
                improved = per_sample > best_loss
                best_loss = torch.where(improved, per_sample, best_loss)
                best_adv[improved] = x_adv[improved]

        if was_training:
            model.train()
        return best_adv.detach()


class FrequencySelectiveAttack(_PGDBase):
    """PGD on STFT magnitude, confined to ``[f_lo, f_hi]`` Hz.

    This is the attack of Eq. (4)-(6). ``f_lo=4000, f_hi=8000`` reproduces the
    band that Table 4 identifies as best for F-SAT training.
    """

    def __init__(
        self,
        stft: BandSelectiveSTFT,
        f_lo: float = 4000.0,
        f_hi: float = 8000.0,
        config: Optional[AttackConfig] = None,
    ) -> None:
        super().__init__(config)
        self.stft = stft
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.stft.band_mask(self.f_lo, self.f_hi, device=x.device, dtype=x.dtype)

    def _budget(self, x: torch.Tensor):
        if self.config.epsilon_mode != "relative":
            return self.config.epsilon, self.config.alpha
        # Per-utterance scale: the mean magnitude inside the attacked band.
        with torch.no_grad():
            magnitude, _ = self.stft.analyze(x)
            mask = self._mask(x).to(magnitude.dtype)
            in_band = (magnitude * mask).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)
            scale = in_band.clamp_min(1e-8).view(-1, 1, 1)
        return self.config.epsilon * scale, self.config.alpha * scale

    def _init_delta(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            magnitude, _ = self.stft.analyze(x)
        delta = torch.zeros_like(magnitude)
        if self.config.random_start:
            delta.uniform_(-1.0, 1.0)
        return delta * self._mask(x)

    def _render(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # inject() applies the band mask itself, and the gradient w.r.t. off-band
        # bins is identically zero, so a delta initialised inside the band stays
        # inside it. The inherited L-infinity clip is therefore sufficient.
        return self.stft.inject(
            x, delta, self.f_lo, self.f_hi, clamp_magnitude=self.config.clamp_magnitude
        )


class TimeDomainAttack(_PGDBase):
    """Ordinary waveform PGD, ``x' = x + delta`` (Section 4.1, "Time domain Attack")."""

    def __init__(self, config: Optional[AttackConfig] = None, clamp_waveform: bool = True) -> None:
        super().__init__(config)
        self.clamp_waveform = clamp_waveform

    def _init_delta(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        if self.config.random_start:
            delta.uniform_(-1.0, 1.0)
        return delta

    def _render(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x + delta
        return x_adv.clamp(-1.0, 1.0) if self.clamp_waveform else x_adv


class PhaseAttack(_PGDBase):
    """PGD on STFT phase, magnitude untouched.

    Included because Section 5.4 / Fig. 8a use it to argue that magnitude is
    the productive attack surface: phase perturbations barely move the
    detector, so F-SAT targets magnitude instead.
    """

    def __init__(
        self,
        stft: BandSelectiveSTFT,
        f_lo: float = 4000.0,
        f_hi: float = 8000.0,
        config: Optional[AttackConfig] = None,
    ) -> None:
        super().__init__(config)
        self.stft = stft
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)

    def _init_delta(self, x: torch.Tensor) -> torch.Tensor:
        # epsilon is an angle in radians here, so relative scaling does not
        # apply: PhaseAttack always uses the absolute budget.
        with torch.no_grad():
            magnitude, _ = self.stft.analyze(x)
        delta = torch.zeros_like(magnitude)
        if self.config.random_start:
            delta.uniform_(-1.0, 1.0)
        return delta * self.stft.band_mask(self.f_lo, self.f_hi, device=x.device, dtype=x.dtype)

    def _render(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            magnitude, phasor = self.stft.analyze(x)
        mask = self.stft.band_mask(self.f_lo, self.f_hi, device=x.device, dtype=magnitude.dtype)
        # Rotate the phasor by a band-limited angle offset, leaving |X| fixed.
        angle = mask * delta
        rotation = torch.polar(torch.ones_like(angle), angle)
        return self.stft.synthesize(magnitude, phasor * rotation, length=x.size(-1))


__all__ = [
    "AttackConfig",
    "FrequencySelectiveAttack",
    "TimeDomainAttack",
    "PhaseAttack",
]
