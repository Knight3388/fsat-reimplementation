"""Evaluation metrics.

Table 2 of arXiv:2411.00121v1 reports accuracy separately for real and fake
audio plus their average, because a detector that collapses onto one class can
still look good on pooled accuracy. EER is added here since it is the standard
anti-spoofing metric and makes results comparable with ASVspoof-style work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

REAL, FAKE = 0, 1


@dataclass
class ClassificationReport:
    """Per-class accuracies in the paper's reporting format."""

    real_accuracy: float
    fake_accuracy: float
    average_accuracy: float
    overall_accuracy: float
    eer: Optional[float] = None
    num_real: int = 0
    num_fake: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        out = {
            "real_accuracy": self.real_accuracy,
            "fake_accuracy": self.fake_accuracy,
            "average_accuracy": self.average_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "num_real": float(self.num_real),
            "num_fake": float(self.num_fake),
        }
        if self.eer is not None:
            out["eer"] = self.eer
        out.update(self.extra)
        return out

    def __str__(self) -> str:
        eer = f"  eer={self.eer * 100:.2f}%" if self.eer is not None else ""
        return (
            f"real={self.real_accuracy * 100:.1f}%  "
            f"fake={self.fake_accuracy * 100:.1f}%  "
            f"avg={self.average_accuracy * 100:.1f}%{eer}"
        )


def compute_eer(scores: np.ndarray, labels: np.ndarray, positive: int = FAKE) -> float:
    """Equal error rate from detection scores.

    Args:
        scores: higher means more likely to belong to ``positive``.
        labels: integer labels.
        positive: which label counts as the positive (target) class.

    Returns:
        EER as a fraction in ``[0, 1]``. Returns ``nan`` if either class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    pos = scores[labels == positive]
    neg = scores[labels != positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_labels = (labels[order] == positive).astype(np.int64)

    # Sweep the threshold from permissive to strict.
    n_pos, n_neg = sorted_labels.sum(), sorted_labels.size - sorted_labels.sum()
    # Misses: positives at or below the threshold. False alarms: negatives above it.
    misses = np.concatenate([[0.0], np.cumsum(sorted_labels)]) / n_pos
    false_alarms = np.concatenate([[float(n_neg)], n_neg - np.cumsum(1 - sorted_labels)]) / n_neg

    idx = np.nanargmin(np.abs(misses - false_alarms))
    return float((misses[idx] + false_alarms[idx]) / 2.0)


def classification_report(
    logits: torch.Tensor, labels: torch.Tensor, with_eer: bool = True
) -> ClassificationReport:
    """Build a :class:`ClassificationReport` from logits and ground-truth labels."""
    logits = logits.detach().float().cpu()
    labels = labels.detach().cpu()
    preds = logits.argmax(dim=1)

    real_mask = labels == REAL
    fake_mask = labels == FAKE
    n_real, n_fake = int(real_mask.sum()), int(fake_mask.sum())

    real_acc = float((preds[real_mask] == REAL).float().mean()) if n_real else float("nan")
    fake_acc = float((preds[fake_mask] == FAKE).float().mean()) if n_fake else float("nan")
    overall = float((preds == labels).float().mean()) if labels.numel() else float("nan")
    average = float(np.nanmean([real_acc, fake_acc]))

    eer = None
    if with_eer and n_real and n_fake:
        scores = torch.softmax(logits, dim=1)[:, FAKE].numpy()
        eer = compute_eer(scores, labels.numpy(), positive=FAKE)

    return ClassificationReport(
        real_accuracy=real_acc,
        fake_accuracy=fake_acc,
        average_accuracy=average,
        overall_accuracy=overall,
        eer=eer,
        num_real=n_real,
        num_fake=n_fake,
    )


__all__ = ["REAL", "FAKE", "ClassificationReport", "compute_eer", "classification_report"]
