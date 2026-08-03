"""End-to-end smoke test on synthetic audio.

Trains three of the Table 3 configurations on procedurally generated audio and
compares them under clean, corrupted and attacked conditions:

    RawNet3                     -- baseline
    RawNet3 + RandAug           -- augmentation only
    RawNet3 + RandAug + F-SAT   -- the full method

This proves the pipeline end to end (the detector must actually learn the
high-frequency artifact planted in the fakes, the attack must degrade it, and
F-SAT must recover some of that loss). It is NOT a reproduction of the paper's
numbers: those require DeepFakeVox-HQ, a full RawNet3, and a real schedule.

Run with:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from fsat import (
    AttackConfig,
    AugmentConfig,
    FSATTrainer,
    RawNet3Detector,
    SyntheticSpeechDataset,
    TimeDomainAttack,
    TrainConfig,
    collate,
    evaluate_corruptions,
)
from fsat.evaluation import evaluate_attack_domains

SR = 16000


def build_loaders(args):
    train_plain = SyntheticSpeechDataset(args.train_items, SR, args.duration, seed=0)
    train_aug = SyntheticSpeechDataset(
        args.train_items, SR, args.duration,
        augment=AugmentConfig(enabled=True, n=2, p=0.5), seed=0,
    )
    test = SyntheticSpeechDataset(args.test_items, SR, args.duration, seed=9999)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    return loader(train_plain, True), loader(train_aug, True), loader(test, False)


def make_trainer(args, adversary: str) -> FSATTrainer:
    torch.manual_seed(0)
    model = RawNet3Detector(
        embed_dim=args.embed_dim,
        channels=args.channels,
        model_scale=4,
        sinc_filters=64,
        sinc_kernel=101,
        sample_rate=SR,
    )
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        adversary=adversary,
        gamma=0.1,          # paper, Fig. 9b
        f_lo=4000.0,        # paper, Table 4
        f_hi=8000.0,
        attack=AttackConfig(epsilon=0.01, alpha=0.004, num_steps=args.pgd_steps),
        n_fft=512,
        hop_length=128,
        sample_rate=SR,
        log_every=0,
        device=args.device,
    )
    return FSATTrainer(model, config)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-items", type=int, default=384)
    p.add_argument("--test-items", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--pgd-steps", type=int, default=2, help="PGD steps during training.")
    p.add_argument(
        "--eval-epsilon", type=float, default=1e-4,
        help="Attack magnitude at EVALUATION time. The paper trains at 0.01 "
             "(Fig. 9a) but evaluates at 1e-4 (Table 5).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train_plain, train_aug, test = build_loaders(args)
    # Table 5 evaluation settings: epsilon = 1e-4 with alpha = 4e-4 in the
    # frequency domain and alpha = 4e-5 in the time domain, over 5 iterations.
    eval_freq = AttackConfig(epsilon=args.eval_epsilon, alpha=args.eval_epsilon * 4, num_steps=5)
    eval_time = AttackConfig(epsilon=args.eval_epsilon, alpha=args.eval_epsilon * 0.4, num_steps=5)

    variants = [
        ("RawNet3", "none", train_plain),
        ("RawNet3+RandAug", "none", train_aug),
        ("RawNet3+RandAug+F-SAT", "fsat", train_aug),
    ]

    results = {}
    for name, adversary, loader in variants:
        print(f"\n=== {name} ===", flush=True)
        started = time.time()
        trainer = make_trainer(args, adversary)
        trainer.fit(loader, test)

        clean = trainer.evaluate(test)
        # Time and frequency attacks use different step sizes in Table 5.
        freq_domains = evaluate_attack_domains(
            trainer.model, test, trainer.stft, 4000, 8000, eval_freq, args.device
        )
        time_report = trainer.evaluate_under_attack(test, TimeDomainAttack(eval_time))
        domains = dict(freq_domains, time=time_report)
        corruptions = evaluate_corruptions(
            trainer.model, test, SR,
            corruptions=["gaussian_noise", "aliasing", "low_pass", "room_simulator", "bit_crush"],
            device=args.device,
        )
        corruption_avg = sum(r.average_accuracy for r in corruptions.values()) / len(corruptions)

        results[name] = {
            "clean": clean.average_accuracy,
            "corruption": corruption_avg,
            "time_attack": domains["time"].average_accuracy,
            "freq_attack": domains["spec_magnitude"].average_accuracy,
            "phase_attack": domains["spec_phase"].average_accuracy,
            "seconds": time.time() - started,
        }
        print(f"  done in {results[name]['seconds']:.0f}s", flush=True)

    header = f"{'Approach':<24}{'Clean':>8}{'Corrupt':>9}{'Atk(t)':>8}{'Atk(f)':>8}{'Atk(ph)':>9}"
    print("\n" + "=" * len(header))
    print("Summary: average of real and fake accuracy (%)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<24}{r['clean']*100:>7.1f}%{r['corruption']*100:>8.1f}%"
            f"{r['time_attack']*100:>7.1f}%{r['freq_attack']*100:>7.1f}%{r['phase_attack']*100:>8.1f}%"
        )
    print("=" * len(header))

    # Mechanism checks. These verify the implementation behaves as the paper
    # describes; they are not a reproduction of its accuracy numbers.
    #
    # F-SAT is compared against RawNet3+RandAug, not against plain RawNet3:
    # the matched control is the model trained on the same augmented data, so
    # the only difference is the adversarial term.
    baseline = results["RawNet3"]
    control = results["RawNet3+RandAug"]
    fsat = results["RawNet3+RandAug+F-SAT"]

    checks = [
        ("baseline learns the task (clean > 60%)", baseline["clean"] > 0.60),
        ("attacks degrade the baseline", baseline["time_attack"] < baseline["clean"]),
        # Fig. 8a: phase attacks are the least effective, which is the paper's
        # stated reason for targeting magnitude instead.
        ("phase attack is weaker than the magnitude attack",
         baseline["phase_attack"] >= baseline["freq_attack"]),
        ("F-SAT >= matched control under the frequency attack",
         fsat["freq_attack"] >= control["freq_attack"]),
        ("F-SAT >= matched control under corruption",
         fsat["corruption"] >= control["corruption"]),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    if not ok:
        print(
            "\n  Note: on this small synthetic task differences of a few points\n"
            "  are within run-to-run noise. Increase --train-items and --epochs\n"
            "  before drawing conclusions from a failure here."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
