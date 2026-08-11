"""F-SAT training and robust evaluation.

Section 4.1 of arXiv:2411.00121v1 defines the objective. After ``K`` PGD
iterations the most perturbed sample is fed back to the detector:

    L_robust(x', y) = H(F_theta(x'), y)
    L_clean (x,  y) = H(F_theta(x),  y)
    L_total         = L_clean + gamma * L_robust

The paper's Fig. 9b selects ``gamma = 0.1`` as the ratio between robust and
clean loss giving the highest overall accuracy, and Fig. 9a selects
``epsilon = 0.01``. Table 4 selects the 4-8 kHz band.

The optimizer and learning-rate schedule are not stated in the paper. Adam
with a cosine schedule is used here as a reasonable default and is fully
configurable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import AttackConfig, FrequencySelectiveAttack, PhaseAttack, TimeDomainAttack, _PGDBase
from .metrics import ClassificationReport, classification_report
from .stft import BandSelectiveSTFT

AdversaryKind = str  # one of: "none", "fsat", "time", "phase"


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Convex combination of a batch with a permutation of itself.

    Mirrors ``regularization.mixup_data`` in the authors' supplementary code:
    a single scalar lambda drawn from Beta(alpha, alpha) for the whole batch.
    ``alpha <= 0`` disables it and returns the batch untouched.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def _mixup_ce(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    """``lam * CE(y_a) + (1 - lam) * CE(y_b)`` -- their ``mixup_criterion``."""
    if lam == 1.0:
        return F.cross_entropy(logits, y_a)
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Defaults marked "(paper)" are the values the paper's ablations select.
    The rest are unspecified in the paper and chosen here as sane defaults.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 5.0

    # --- F-SAT objective -------------------------------------------------
    adversary: AdversaryKind = "fsat"
    gamma: float = 0.1  # (paper, Fig. 9b) weight on L_robust
    f_lo: float = 4000.0  # (paper, Table 4) best band lower edge
    f_hi: float = 8000.0  # (paper, Table 4) best band upper edge
    attack: AttackConfig = field(default_factory=lambda: AttackConfig(epsilon=0.01, alpha=0.004, num_steps=5))

    # --- STFT front end --------------------------------------------------
    n_fft: int = 1024
    hop_length: int = 256
    sample_rate: int = 16000

    device: str = "cpu"
    log_every: int = 10
    scheduler: bool = True

    # Warmup + cosine, matching the schedule in the authors' released
    # supplementary code (lr_schedule.WarmUpCosineLR). Set warmup_epochs=0 for
    # the plain cosine schedule.
    warmup_epochs: int = 0
    warmup_lr: float = 1e-6
    min_lr: float = 0.0

    # Mixup alpha (0 disables). The authors' train_attack_frequency.py opens with
    # `assert args.mixup, "Mixup is preferred for spectrum attack"` -- mixup is
    # MANDATORY for the F-SAT attack in their code, not an optional extra, and
    # their reproduce command passes --mixup --mixup_alpha 0.5.
    mixup_alpha: float = 0.0


class FSATTrainer:
    """Trains a waveform detector with the F-SAT min-max objective.

    Set ``config.adversary = "none"`` to train the plain baseline (the
    ``RawNet3`` / ``RawNet3+RandAug`` rows of Table 3), ``"time"`` for
    time-domain adversarial training (the ``+AT(Time)`` row), and ``"fsat"``
    for the full method.
    """

    def __init__(self, model: nn.Module, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.stft = BandSelectiveSTFT(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            sample_rate=self.config.sample_rate,
        ).to(self.device)
        self.attack = self._build_attack(self.config.adversary)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ #
    def _build_scheduler(self):
        cfg = self.config
        if not cfg.scheduler:
            return None
        total = max(1, cfg.epochs)
        if cfg.warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total)

        warm, base = cfg.warmup_epochs, cfg.learning_rate

        def lr_lambda(epoch: int) -> float:
            # Returns a MULTIPLIER on the optimizer's base lr, so every branch
            # is expressed relative to `base`.
            if epoch < warm:
                lr = cfg.warmup_lr + (base - cfg.warmup_lr) * (epoch / max(warm, 1))
            else:
                import math
                progress = (epoch - warm) / max(total - warm, 1)
                cos_out = math.cos(math.pi * progress) + 1
                lr = cfg.min_lr + 0.5 * (base - cfg.min_lr) * cos_out
            return lr / base if base else 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------ #
    def _build_attack(self, kind: AdversaryKind) -> Optional[_PGDBase]:
        cfg = self.config
        if kind == "none":
            return None
        if kind == "fsat":
            return FrequencySelectiveAttack(self.stft, cfg.f_lo, cfg.f_hi, cfg.attack)
        if kind == "time":
            return TimeDomainAttack(cfg.attack)
        if kind == "phase":
            return PhaseAttack(self.stft, cfg.f_lo, cfg.f_hi, cfg.attack)
        raise ValueError(f"unknown adversary {kind!r}; expected none|fsat|time|phase")

    # ------------------------------------------------------------------ #
    def train_epoch(self, loader: Iterable) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "loss_clean": 0.0, "loss_robust": 0.0, "correct": 0.0, "count": 0.0}

        for step, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Inner maximization. Generated with the model in eval mode so the
            # attack does not pollute BatchNorm running statistics.
            x_adv = self.attack(self.model, x, y) if self.attack is not None else None

            # Outer minimization. Note the ordering the authors use: the attack
            # is crafted on the ORIGINAL inputs, and mixup is then applied
            # separately to the clean and adversarial branches with INDEPENDENT
            # lambdas and permutations.
            self.optimizer.zero_grad(set_to_none=True)
            a = self.config.mixup_alpha

            x_c, ya_c, yb_c, lam_c = _mixup(x, y, a)
            logits_clean = self.model(x_c)
            loss_clean = _mixup_ce(logits_clean, ya_c, yb_c, lam_c)

            if x_adv is not None:
                x_a, ya_a, yb_a, lam_a = _mixup(x_adv, y, a)
                loss_robust = _mixup_ce(self.model(x_a), ya_a, yb_a, lam_a)
                loss = loss_clean + self.config.gamma * loss_robust
            else:
                loss_robust = torch.zeros((), device=self.device)
                loss = loss_clean

            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            batch = y.size(0)
            loss_value = float(loss.detach())
            clean_value = float(loss_clean.detach())
            robust_value = float(loss_robust.detach())

            totals["loss"] += loss_value * batch
            totals["loss_clean"] += clean_value * batch
            totals["loss_robust"] += robust_value * batch
            totals["correct"] += float((logits_clean.detach().argmax(1) == y).sum())
            totals["count"] += batch

            if self.config.log_every and step % self.config.log_every == 0:
                print(
                    f"    step {step:4d}  loss={loss_value:.4f}  "
                    f"clean={clean_value:.4f}  robust={robust_value:.4f}",
                    flush=True,
                )

        n = max(totals["count"], 1.0)
        return {
            "loss": totals["loss"] / n,
            "loss_clean": totals["loss_clean"] / n,
            "loss_robust": totals["loss_robust"] / n,
            "accuracy": totals["correct"] / n,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, loader: Iterable):
        """Raw ``(logits, labels)`` over a loader, in loader order.

        Kept separate from :meth:`evaluate` so callers that need per-utterance
        scores (t-DCF, per-attack breakdowns) can get them without re-running
        the forward pass. Order is preserved, so a caller may zip the result
        against an unshuffled dataset's items to recover utterance ids.
        """
        self.model.eval()
        logits, labels = [], []
        for x, y in loader:
            logits.append(self.model(x.to(self.device)).cpu())
            labels.append(y)
        return torch.cat(logits), torch.cat(labels)

    def evaluate(self, loader: Iterable) -> ClassificationReport:
        """Clean accuracy, in the paper's real/fake/avg format."""
        logits, labels = self.predict(loader)
        return classification_report(logits, labels)

    def evaluate_under_attack(self, loader: Iterable, attack: _PGDBase) -> ClassificationReport:
        """Accuracy under an adversarial attack (Table 5 / Fig. 7b).

        Pass any attack instance. To reproduce a *transfer* (black-box) setting,
        build the attack against a surrogate model and call
        :meth:`evaluate_under_attack_from` instead.
        """
        return self.evaluate_under_attack_from(loader, attack, source_model=self.model)

    def evaluate_under_attack_from(
        self, loader: Iterable, attack: _PGDBase, source_model: nn.Module
    ) -> ClassificationReport:
        """Evaluate ``self.model`` on examples crafted against ``source_model``.

        ``source_model is self.model`` gives the white-box case (Table 5,
        source ``A``); a different model with the same architecture gives the
        grey-box case (``A'``); a different architecture gives black-box (``B``).
        """
        self.model.eval()
        source_model.eval()
        logits, labels = [], []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            x_adv = attack(source_model, x, y)
            with torch.no_grad():
                logits.append(self.model(x_adv).cpu())
            labels.append(y.cpu())
        return classification_report(torch.cat(logits), torch.cat(labels))

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """Run the full training loop, returning per-epoch metrics."""
        for epoch in range(1, self.config.epochs + 1):
            started = time.time()
            print(f"  epoch {epoch}/{self.config.epochs}", flush=True)
            metrics = self.train_epoch(train_loader)
            metrics["epoch"] = epoch
            metrics["seconds"] = time.time() - started

            if val_loader is not None:
                report = self.evaluate(val_loader)
                metrics.update({f"val_{k}": v for k, v in report.as_dict().items()})
                print(f"    val  {report}", flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history.append(metrics)
            if on_epoch_end is not None:
                on_epoch_end(epoch, metrics)
        return self.history

    # ------------------------------------------------------------------ #
    def band_attack(self, f_lo: float, f_hi: float, config: Optional[AttackConfig] = None) -> FrequencySelectiveAttack:
        """Convenience builder for a band-limited attack, for Table 4 sweeps."""
        return FrequencySelectiveAttack(self.stft, f_lo, f_hi, config or self.config.attack)


__all__ = ["TrainConfig", "FSATTrainer"]
