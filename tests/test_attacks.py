"""Tests for the PGD attacks of Section 4.1."""

import pytest
import torch
import torch.nn.functional as F

from fsat.attacks import AttackConfig, FrequencySelectiveAttack, PhaseAttack, TimeDomainAttack
from fsat.stft import BandSelectiveSTFT

SR = 16000


@pytest.fixture
def stft():
    return BandSelectiveSTFT(n_fft=512, hop_length=128, sample_rate=SR)


@pytest.fixture
def config():
    return AttackConfig(epsilon=0.01, alpha=0.004, num_steps=3, num_restarts=1)


def _all_attacks(stft, config):
    return {
        "fsat": FrequencySelectiveAttack(stft, 4000, 8000, config),
        "time": TimeDomainAttack(config),
        "phase": PhaseAttack(stft, 4000, 8000, config),
    }


@pytest.mark.parametrize("name", ["fsat", "time", "phase"])
def test_shape_and_finiteness_preserved(tiny_detector, batch, stft, config, name):
    x, y = batch
    x_adv = _all_attacks(stft, config)[name](tiny_detector, x, y)
    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()
    assert not x_adv.requires_grad


@pytest.mark.parametrize("name", ["fsat", "time", "phase"])
def test_attack_increases_loss(tiny_detector, batch, stft, config, name):
    """PGD must make the model worse, otherwise it is not solving the inner max."""
    x, y = batch
    x_adv = _all_attacks(stft, config)[name](tiny_detector, x, y)
    with torch.no_grad():
        clean = F.cross_entropy(tiny_detector(x), y)
        adv = F.cross_entropy(tiny_detector(x_adv), y)
    assert float(adv) >= float(clean)


def test_time_domain_respects_epsilon_ball(tiny_detector, batch, config):
    x, y = batch
    x_adv = TimeDomainAttack(config, clamp_waveform=False)(tiny_detector, x, y)
    assert float((x_adv - x).abs().max()) <= config.epsilon + 1e-6


def test_zero_steps_and_no_random_start_is_identity(tiny_detector, batch, stft):
    cfg = AttackConfig(epsilon=0.01, alpha=0.0, num_steps=1, random_start=False)
    x, y = batch
    x_adv = FrequencySelectiveAttack(stft, 4000, 8000, cfg)(tiny_detector, x, y)
    assert torch.allclose(x_adv, x, atol=1e-4)


def test_more_steps_do_not_reduce_attack_strength(tiny_detector, batch, stft):
    """Attack success should be monotone in PGD budget, a basic sanity check."""
    x, y = batch
    losses = []
    for steps in (1, 5):
        cfg = AttackConfig(epsilon=0.02, alpha=0.008, num_steps=steps, random_start=False)
        x_adv = FrequencySelectiveAttack(stft, 4000, 8000, cfg)(tiny_detector, x, y)
        with torch.no_grad():
            losses.append(float(F.cross_entropy(tiny_detector(x_adv), y)))
    assert losses[1] >= losses[0] - 1e-4


def test_restarts_pick_the_worst_case(tiny_detector, batch, stft):
    x, y = batch
    base = AttackConfig(epsilon=0.02, alpha=0.008, num_steps=2, num_restarts=1)
    many = AttackConfig(epsilon=0.02, alpha=0.008, num_steps=2, num_restarts=4)

    torch.manual_seed(0)
    one = FrequencySelectiveAttack(stft, 4000, 8000, base)(tiny_detector, x, y)
    torch.manual_seed(0)
    four = FrequencySelectiveAttack(stft, 4000, 8000, many)(tiny_detector, x, y)

    with torch.no_grad():
        loss_one = F.cross_entropy(tiny_detector(one), y, reduction="none")
        loss_four = F.cross_entropy(tiny_detector(four), y, reduction="none")
    assert bool((loss_four >= loss_one - 1e-5).all())


def test_model_training_mode_is_restored(tiny_detector, batch, stft, config):
    x, y = batch
    tiny_detector.train()
    try:
        FrequencySelectiveAttack(stft, 4000, 8000, config)(tiny_detector, x, y)
        assert tiny_detector.training is True
    finally:
        tiny_detector.eval()


def test_attack_leaves_model_gradients_untouched(tiny_detector, batch, stft, config):
    """The inner maximization must not accumulate parameter gradients."""
    x, y = batch
    tiny_detector.zero_grad(set_to_none=True)
    FrequencySelectiveAttack(stft, 4000, 8000, config)(tiny_detector, x, y)
    assert all(p.grad is None for p in tiny_detector.parameters())


@pytest.mark.parametrize("f_lo,f_hi", [(0, 8000), (2000, 8000), (4000, 8000), (6000, 8000)])
def test_all_paper_bands_run(tiny_detector, batch, stft, config, f_lo, f_hi):
    x, y = batch
    assert FrequencySelectiveAttack(stft, f_lo, f_hi, config)(tiny_detector, x, y).shape == x.shape


def test_relative_epsilon_scales_with_input_gain(tiny_detector, stft):
    """Relative mode must give gain-invariant attack strength.

    Absolute mode does not: the same epsilon is a large perturbation on a quiet
    clip and a negligible one on a loud clip.
    """
    torch.manual_seed(0)
    quiet = torch.randn(2, SR) * 0.01
    loud = quiet * 50.0
    y = torch.tensor([0, 1])

    def relative_strength(x, mode):
        cfg = AttackConfig(
            epsilon=0.05, alpha=0.02, num_steps=2, random_start=False, epsilon_mode=mode
        )
        x_adv = FrequencySelectiveAttack(stft, 4000, 8000, cfg)(tiny_detector, x, y)
        return float((x_adv - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt())

    rel_quiet = relative_strength(quiet, "relative")
    rel_loud = relative_strength(loud, "relative")
    assert rel_quiet == pytest.approx(rel_loud, rel=0.15)

    abs_quiet = relative_strength(quiet, "absolute")
    abs_loud = relative_strength(loud, "absolute")
    assert abs_loud < abs_quiet / 5.0


def test_absolute_mode_is_the_default():
    assert AttackConfig().epsilon_mode == "absolute"


def test_phase_attack_preserves_magnitude(tiny_detector, batch, stft, config):
    """The phase arm must leave |X| alone, that is what distinguishes it."""
    x, y = batch
    x_adv = PhaseAttack(stft, 0, 8000, config)(tiny_detector, x, y)
    mag_before, _ = stft.analyze(x)
    mag_after, _ = stft.analyze(x_adv)
    # Overlap-add of rotated frames is not exactly magnitude-preserving, but
    # total spectral energy must stay close.
    ratio = float(mag_after.pow(2).sum() / mag_before.pow(2).sum())
    assert 0.9 < ratio < 1.1
