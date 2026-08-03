"""F-SAT: Frequency-Selective Adversarial Training for deepfake audio detection.

Reimplementation of Zhang et al., "I Can Hear You: Selective Robust Training
for Deepfake Audio Detection" (arXiv:2411.00121v1).

Quick start::

    from fsat import RawNet3Detector, FSATTrainer, TrainConfig
    from fsat.data import SyntheticSpeechDataset, AugmentConfig, collate
    from torch.utils.data import DataLoader

    train = SyntheticSpeechDataset(256, augment=AugmentConfig(enabled=True, n=2, p=0.5))
    loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=collate)

    model = RawNet3Detector()
    trainer = FSATTrainer(model, TrainConfig(adversary="fsat", gamma=0.1))
    trainer.fit(loader)
"""

from .attacks import AttackConfig, FrequencySelectiveAttack, PhaseAttack, TimeDomainAttack
from .data import AugmentConfig, ManifestDataset, SyntheticSpeechDataset, collate
from .evaluation import evaluate_attack_bands, evaluate_attack_domains, evaluate_corruptions
from .metrics import ClassificationReport, classification_report, compute_eer
from .models import RawNet3, RawNet3Detector
from .randaugment import CORRUPTION_SUITE, RANDAUGMENT_OPS, apply_transform, rand_augment
from .stft import BandSelectiveSTFT
from .trainer import FSATTrainer, TrainConfig

__version__ = "0.1.0"

__all__ = [
    # core
    "BandSelectiveSTFT",
    "AttackConfig",
    "FrequencySelectiveAttack",
    "TimeDomainAttack",
    "PhaseAttack",
    "FSATTrainer",
    "TrainConfig",
    # models
    "RawNet3",
    "RawNet3Detector",
    # augmentation
    "rand_augment",
    "apply_transform",
    "RANDAUGMENT_OPS",
    "CORRUPTION_SUITE",
    # data
    "SyntheticSpeechDataset",
    "ManifestDataset",
    "AugmentConfig",
    "collate",
    # metrics and evaluation
    "ClassificationReport",
    "classification_report",
    "compute_eer",
    "evaluate_corruptions",
    "evaluate_attack_bands",
    "evaluate_attack_domains",
]
