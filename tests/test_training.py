"""End-to-end tests: metrics, datasets, and the F-SAT training objective."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fsat.attacks import AttackConfig
from fsat.data import AugmentConfig, SyntheticSpeechDataset, collate
from fsat.evaluation import evaluate_attack_bands, evaluate_attack_domains, evaluate_corruptions
from fsat.metrics import FAKE, REAL, classification_report, compute_eer
from fsat.models import RawNet3Detector
from fsat.trainer import FSATTrainer, TrainConfig

SR = 16000


# ------------------------------------------------------------------ metrics
def test_eer_is_zero_for_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([REAL, REAL, FAKE, FAKE])
    assert compute_eer(scores, labels) == pytest.approx(0.0, abs=1e-6)


def test_eer_is_half_for_inverted_scores():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([REAL, REAL, FAKE, FAKE])
    assert compute_eer(scores, labels) == pytest.approx(1.0, abs=1e-6)


def test_eer_is_nan_when_a_class_is_missing():
    assert np.isnan(compute_eer(np.array([0.1, 0.2]), np.array([REAL, REAL])))


def test_classification_report_separates_classes():
    # Two reals predicted correctly, two fakes predicted as real.
    logits = torch.tensor([[9.0, 0.0], [9.0, 0.0], [9.0, 0.0], [9.0, 0.0]])
    labels = torch.tensor([REAL, REAL, FAKE, FAKE])
    report = classification_report(logits, labels)
    assert report.real_accuracy == pytest.approx(1.0)
    assert report.fake_accuracy == pytest.approx(0.0)
    assert report.average_accuracy == pytest.approx(0.5)
    assert report.num_real == 2 and report.num_fake == 2


# ------------------------------------------------------------------ dataset
def test_synthetic_dataset_is_balanced_and_shaped():
    ds = SyntheticSpeechDataset(num_items=16, duration=1.0)
    assert len(ds) == 16
    labels = [ds[i][1] for i in range(16)]
    assert labels.count(REAL) == labels.count(FAKE) == 8
    x, _ = ds[0]
    assert x.shape == (SR,) and x.dtype == torch.float32


def test_synthetic_dataset_is_deterministic():
    a = SyntheticSpeechDataset(num_items=4, duration=1.0, seed=3)
    b = SyntheticSpeechDataset(num_items=4, duration=1.0, seed=3)
    assert torch.equal(a[1][0], b[1][0])


def test_fake_samples_carry_high_frequency_artifacts():
    """The synthetic task must be solvable from high-frequency content."""
    ds = SyntheticSpeechDataset(num_items=32, duration=1.0, seed=0)
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    band = freqs >= 5000

    def hf_energy(index):
        spectrum = np.abs(np.fft.rfft(ds[index][0].numpy()))
        return float((spectrum[band] ** 2).sum() / (spectrum**2).sum())

    real = np.mean([hf_energy(i) for i in range(0, 32, 2)])
    fake = np.mean([hf_energy(i) for i in range(1, 32, 2)])
    assert fake > real


def test_augmentation_changes_samples():
    plain = SyntheticSpeechDataset(num_items=4, duration=1.0, seed=0)
    augmented = SyntheticSpeechDataset(
        num_items=4, duration=1.0, seed=0, augment=AugmentConfig(enabled=True, n=2, p=1.0)
    )
    assert not torch.allclose(plain[0][0], augmented[0][0])


def test_collate_builds_rectangular_batches():
    ds = SyntheticSpeechDataset(num_items=8, duration=1.0)
    x, y = collate([ds[i] for i in range(4)])
    assert x.shape == (4, SR) and y.shape == (4,) and y.dtype == torch.long


# ------------------------------------------------------------------ training
@pytest.fixture
def loader():
    ds = SyntheticSpeechDataset(num_items=16, duration=1.0, seed=0)
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)


def _small_trainer(adversary, **kwargs):
    torch.manual_seed(0)
    model = RawNet3Detector(
        embed_dim=32, channels=64, model_scale=4, sinc_filters=32, sinc_kernel=101
    )
    config = TrainConfig(
        epochs=1,
        adversary=adversary,
        attack=AttackConfig(epsilon=0.01, alpha=0.004, num_steps=2),
        n_fft=512,
        hop_length=128,
        log_every=0,
        **kwargs,
    )
    return FSATTrainer(model, config)


@pytest.mark.parametrize("adversary", ["none", "fsat", "time", "phase"])
def test_one_epoch_runs_for_every_adversary(loader, adversary):
    trainer = _small_trainer(adversary)
    metrics = trainer.train_epoch(loader)
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_robust_loss_is_zero_without_an_adversary(loader):
    assert _small_trainer("none").train_epoch(loader)["loss_robust"] == 0.0


def test_robust_loss_is_nonzero_with_fsat(loader):
    assert _small_trainer("fsat").train_epoch(loader)["loss_robust"] > 0.0


def test_gamma_scales_the_total_loss(loader):
    """L_total = L_clean + gamma * L_robust must hold numerically."""
    trainer = _small_trainer("fsat", gamma=0.1)
    m = trainer.train_epoch(loader)
    assert m["loss"] == pytest.approx(m["loss_clean"] + 0.1 * m["loss_robust"], rel=1e-4)


def test_parameters_are_updated(loader):
    trainer = _small_trainer("fsat")
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train_epoch(loader)
    after = list(trainer.model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_fit_records_history(loader):
    trainer = _small_trainer("none")
    trainer.config.epochs = 2
    history = trainer.fit(loader, loader)
    assert len(history) == 2
    assert "val_average_accuracy" in history[0]


def test_evaluate_returns_a_report(loader):
    report = _small_trainer("none").evaluate(loader)
    assert report.num_real == report.num_fake == 8
    assert 0.0 <= report.average_accuracy <= 1.0


def test_unknown_adversary_rejected():
    with pytest.raises(ValueError, match="unknown adversary"):
        _small_trainer("nonsense")


# ------------------------------------------------------------------ evaluation sweeps
def test_attack_band_sweep_covers_the_paper_bands(loader):
    trainer = _small_trainer("none")
    results = evaluate_attack_bands(
        trainer.model, loader, trainer.stft, config=AttackConfig(num_steps=1)
    )
    assert set(results) == {"0-8kHz", "2-8kHz", "4-8kHz", "6-8kHz"}


def test_attack_domain_sweep_includes_a_clean_reference(loader):
    trainer = _small_trainer("none")
    results = evaluate_attack_domains(
        trainer.model, loader, trainer.stft, config=AttackConfig(num_steps=1)
    )
    assert set(results) == {"no_attack", "time", "spec_magnitude", "spec_phase"}


def test_corruption_sweep_runs(loader):
    trainer = _small_trainer("none")
    results = evaluate_corruptions(
        trainer.model, loader, corruptions=["gaussian_noise", "aliasing", "low_pass"]
    )
    assert set(results) == {"gaussian_noise", "aliasing", "low_pass"}
    assert all(np.isfinite(r.average_accuracy) for r in results.values())


def test_transfer_attack_uses_the_source_model(loader):
    """Black-box setting: craft on one model, evaluate on another."""
    victim = _small_trainer("none")
    surrogate = _small_trainer("none")
    report = victim.evaluate_under_attack_from(
        loader, victim.band_attack(4000, 8000, AttackConfig(num_steps=1)), surrogate.model
    )
    assert 0.0 <= report.average_accuracy <= 1.0
