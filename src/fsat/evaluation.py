"""Robustness evaluation sweeps.

Reproduces the shape of the paper's robustness results:

* :func:`evaluate_corruptions` -- accuracy across the 24-corruption battery of
  Fig. 7a.
* :func:`evaluate_attack_bands` -- accuracy under frequency-selective attacks in
  the 0-8k, 2-8k, 4-8k and 6-8k bands, the columns of Table 4 and Table 5.
* :func:`evaluate_attack_domains` -- accuracy under time, magnitude and phase
  attacks, the ablation of Fig. 8a and Table 3.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .attacks import AttackConfig, FrequencySelectiveAttack, PhaseAttack, TimeDomainAttack
from .metrics import ClassificationReport, classification_report
from .randaugment import CORRUPTION_SUITE, apply_transform
from .stft import BandSelectiveSTFT

#: The four attack bands tabulated in Table 4 / Table 5, in Hz.
ATTACK_BANDS: List[Tuple[float, float]] = [
    (0.0, 8000.0),
    (2000.0, 8000.0),
    (4000.0, 8000.0),
    (6000.0, 8000.0),
]


@torch.no_grad()
def evaluate_corruptions(
    model: nn.Module,
    loader: Iterable,
    sample_rate: int = 16000,
    corruptions: Optional[Sequence[str]] = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Dict[str, ClassificationReport]:
    """Accuracy under each corruption in the battery (Fig. 7a).

    Corruptions are applied on CPU with numpy, matching how they would occur in
    transmission or recording, then the corrupted batch is scored.

    Note: the loader is materialised into memory once so that every corruption
    is measured on byte-identical audio. On a large test set, pass a subset
    loader rather than the full evaluation set.
    """
    device = torch.device(device)
    model.eval()
    names = list(corruptions) if corruptions is not None else CORRUPTION_SUITE
    results: Dict[str, ClassificationReport] = {}

    # Materialise once so every corruption sees identical audio.
    batches = [(x.cpu(), y.cpu()) for x, y in loader]

    for name in names:
        rng = np.random.default_rng(seed)
        logits, labels = [], []
        for x, y in batches:
            corrupted = np.stack(
                [apply_transform(name, x[i].numpy(), sample_rate, rng) for i in range(x.size(0))]
            )
            batch = torch.from_numpy(corrupted).to(device)
            logits.append(model(batch).cpu())
            labels.append(y)
        results[name] = classification_report(torch.cat(logits), torch.cat(labels))
    return results


def evaluate_attack_bands(
    model: nn.Module,
    loader: Iterable,
    stft: BandSelectiveSTFT,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    config: Optional[AttackConfig] = None,
    device: str | torch.device = "cpu",
    source_model: Optional[nn.Module] = None,
) -> Dict[str, ClassificationReport]:
    """Accuracy under band-limited magnitude attacks (Table 4, Table 5).

    ``source_model`` defaults to ``model`` (white-box). Pass a different model
    to measure transfer.
    """
    device = torch.device(device)
    model.eval()
    source = source_model or model
    source.eval()
    results: Dict[str, ClassificationReport] = {}

    for f_lo, f_hi in bands or ATTACK_BANDS:
        attack = FrequencySelectiveAttack(stft, f_lo, f_hi, config or AttackConfig())
        logits, labels = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = attack(source, x, y)
            with torch.no_grad():
                logits.append(model(x_adv).cpu())
            labels.append(y.cpu())
        key = f"{int(f_lo / 1000)}-{int(f_hi / 1000)}kHz"
        results[key] = classification_report(torch.cat(logits), torch.cat(labels))
    return results


def evaluate_attack_domains(
    model: nn.Module,
    loader: Iterable,
    stft: BandSelectiveSTFT,
    f_lo: float = 4000.0,
    f_hi: float = 8000.0,
    config: Optional[AttackConfig] = None,
    device: str | torch.device = "cpu",
    time_config: Optional[AttackConfig] = None,
) -> Dict[str, ClassificationReport]:
    """Accuracy under time / magnitude / phase attacks (Fig. 8a).

    The paper's finding is that magnitude attacks degrade the detector most and
    phase attacks least, which is why F-SAT perturbs magnitude.

    ``time_config`` overrides the budget for the time-domain probe only. The
    paper uses a different step size in the time domain (alpha 4e-5) than in
    frequency (4e-4), and applying one budget to both is not a like-for-like
    comparison: a perturbation that is modest on an STFT magnitude is large on
    a waveform in [-1, 1].
    """
    device = torch.device(device)
    model.eval()
    cfg = config or AttackConfig()
    attacks = {
        "no_attack": None,
        "time": TimeDomainAttack(time_config or cfg),
        "spec_magnitude": FrequencySelectiveAttack(stft, f_lo, f_hi, cfg),
        "spec_phase": PhaseAttack(stft, f_lo, f_hi, cfg),
    }

    results: Dict[str, ClassificationReport] = {}
    for name, attack in attacks.items():
        logits, labels = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch = x if attack is None else attack(model, x, y)
            with torch.no_grad():
                logits.append(model(batch).cpu())
            labels.append(y.cpu())
        results[name] = classification_report(torch.cat(logits), torch.cat(labels))
    return results


__all__ = [
    "ATTACK_BANDS",
    "evaluate_corruptions",
    "evaluate_attack_bands",
    "evaluate_attack_domains",
]
