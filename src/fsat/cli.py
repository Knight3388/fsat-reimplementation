"""Command-line entrypoint for training and evaluating F-SAT models.

Examples::

    # Baseline RawNet3, no augmentation, no adversarial training
    fsat-train --adversary none --epochs 5

    # RawNet3 + RandAug (the Table 3 middle row)
    fsat-train --adversary none --randaug --epochs 5

    # RawNet3 + RandAug + F-SAT (the full method)
    fsat-train --adversary fsat --randaug --gamma 0.1 --f-lo 4000 --f-hi 8000

    # Train on a real corpus
    fsat-train --train-manifest train.tsv --val-manifest test.tsv --randaug --adversary fsat
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .attacks import AttackConfig
from .data import AugmentConfig, ManifestDataset, SyntheticSpeechDataset, collate
from .evaluation import evaluate_attack_bands, evaluate_attack_domains, evaluate_corruptions
from .models import RawNet3Detector
from .trainer import FSATTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fsat-train",
        description="Train a deepfake audio detector with Frequency-Selective Adversarial Training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--train-manifest", help="TSV of 'path<TAB>label'. Omit to use synthetic audio.")
    data.add_argument("--val-manifest", help="Validation manifest, used for model selection.")
    data.add_argument(
        "--test-manifest",
        help="Held-out manifest for final reporting. Keeping this separate from "
             "--val-manifest is what stops checkpoint selection from peeking at "
             "the evaluation set. Falls back to --val-manifest if omitted.",
    )
    data.add_argument("--root", help="Prefix for relative paths in the manifests.")
    data.add_argument("--sample-rate", type=int, default=16000)
    data.add_argument("--duration", type=float, default=3.0, help="Seconds per training crop.")
    data.add_argument("--train-items", type=int, default=256, help="Synthetic training set size.")
    data.add_argument("--val-items", type=int, default=64, help="Synthetic validation set size.")
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--workers", type=int, default=0)
    data.add_argument(
        "--balanced-sampler", action="store_true",
        help="Sample classes with equal probability. Required on skewed corpora "
             "such as ASVspoof2019 LA (8.8:1 spoof:bonafide), where the minority "
             "class otherwise collapses to 0%% accuracy.",
    )

    aug = p.add_argument_group("RandAugment (Section 4.2)")
    aug.add_argument("--randaug", action="store_true", help="Enable audio RandAugment on the training split.")
    aug.add_argument("--randaug-n", type=int, default=2, help="Transforms drawn per sample (N).")
    aug.add_argument("--randaug-p", type=float, default=0.5, help="Probability of augmenting (p).")

    adv = p.add_argument_group("F-SAT (Section 4.1)")
    adv.add_argument("--adversary", choices=["none", "fsat", "time", "phase"], default="fsat")
    adv.add_argument("--gamma", type=float, default=0.1, help="Weight on L_robust (paper: 0.1).")
    adv.add_argument("--f-lo", type=float, default=4000.0, help="Band lower edge, Hz (paper: 4000).")
    adv.add_argument("--f-hi", type=float, default=8000.0, help="Band upper edge, Hz (paper: 8000).")
    adv.add_argument("--epsilon", type=float, default=0.01, help="Attack magnitude (paper: 0.01).")
    adv.add_argument("--alpha", type=float, default=0.004, help="PGD step size.")
    adv.add_argument("--steps", type=int, default=5, help="PGD iterations (K).")
    adv.add_argument("--restarts", type=int, default=1)

    ev = p.add_argument_group("evaluation-time attack budget (Table 5)")
    ev.add_argument("--load", help="Load model weights from this checkpoint before running.")
    ev.add_argument("--eval-only", action="store_true", help="Skip training; evaluate --load only.")
    ev.add_argument(
        "--eval-epsilon", type=float, default=None,
        help="Attack magnitude for the evaluation sweeps (paper Table 5: 1e-4). "
             "Defaults to --epsilon, which evaluates at the TRAINING budget and "
             "flatters a model whose training attack matches the probe.",
    )
    ev.add_argument("--eval-alpha", type=float, default=None,
                    help="PGD step for frequency-domain evaluation (paper: 4e-4). Defaults to --alpha.")
    ev.add_argument("--eval-alpha-time", type=float, default=None,
                    help="PGD step for the time-domain probe (paper: 4e-5). Defaults to --eval-alpha.")

    opt = p.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=10)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    out = p.add_argument_group("output and evaluation")
    out.add_argument("--save", help="Path to write the trained checkpoint.")
    out.add_argument(
        "--save-best", action="store_true",
        help="Write --save at the best validation epoch (lowest EER) instead of "
             "the last one, and evaluate that checkpoint. Needs --val-manifest.",
    )
    out.add_argument("--report", help="Path to write a JSON metrics report.")
    out.add_argument("--eval-corruptions", action="store_true", help="Run the 24-corruption battery (Fig. 7a).")
    out.add_argument("--eval-attacks", action="store_true", help="Run the band and domain attack sweeps (Tables 3-5).")
    out.add_argument(
        "--eval-subset", type=int, default=0,
        help="Cap the attack and corruption sweeps at this many utterances (0 = no cap). "
             "Clean accuracy always uses the full set. The cap is recorded in the report.",
    )
    out.add_argument("--log-every", type=int, default=10)
    return p


def _dataset_labels(dataset) -> list:
    """Per-item labels, without decoding any audio."""
    if hasattr(dataset, "items"):  # ManifestDataset
        return [label for _, label in dataset.items]
    if hasattr(dataset, "labels"):  # SyntheticSpeechDataset
        return list(dataset.labels)
    raise TypeError(f"cannot read labels from {type(dataset).__name__}")


def _balanced_sampler(dataset) -> WeightedRandomSampler:
    """Draw each class with equal probability, with replacement.

    ASVspoof2019 LA train is 2,580 bonafide against 22,800 spoof. Trained
    unweighted, the detector reaches ~90% pooled accuracy by predicting "spoof"
    everywhere while real-class accuracy goes to zero, which is precisely the
    failure the paper's real/fake/average reporting format exists to expose.
    """
    labels = _dataset_labels(dataset)
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for label in labels]
    print(
        "balanced sampler: "
        + ", ".join(f"class {k}={v}" for k, v in sorted(counts.items())),
        flush=True,
    )
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    if args.workers > 0:
        # The default file_descriptor sharing strategy holds one FD per shared
        # tensor. Evaluating a large manifest (ASVspoof2019 eval is 71,237
        # utterances) exhausts a 1024 FD limit and kills the run after training
        # has already finished. file_system does not accumulate descriptors.
        torch.multiprocessing.set_sharing_strategy("file_system")

    augment = AugmentConfig(enabled=args.randaug, n=args.randaug_n, p=args.randaug_p)

    if args.train_manifest:
        train_set = ManifestDataset(
            args.train_manifest, args.sample_rate, args.duration, augment, root=args.root, seed=args.seed
        )
        val_set = (
            ManifestDataset(
                args.val_manifest, args.sample_rate, args.duration, None,
                random_crop=False, root=args.root, seed=args.seed + 1,
            )
            if args.val_manifest
            else None
        )
        test_set = (
            ManifestDataset(
                args.test_manifest, args.sample_rate, args.duration, None,
                random_crop=False, root=args.root, seed=args.seed + 2,
            )
            if args.test_manifest
            else None
        )
    else:
        test_set = None
        print("no --train-manifest given, using synthetic audio", file=sys.stderr)
        train_set = SyntheticSpeechDataset(
            args.train_items, args.sample_rate, args.duration, augment, seed=args.seed
        )
        val_set = SyntheticSpeechDataset(
            args.val_items, args.sample_rate, args.duration, None, seed=args.seed + 1000
        )

    sampler = _balanced_sampler(train_set) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate, num_workers=args.workers, drop_last=True,
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                   collate_fn=collate, num_workers=args.workers)
        if val_set is not None
        else None
    )

    def _eval_loader(dataset, limit: int = 0):
        if dataset is None:
            return None
        if limit and limit < len(dataset):
            # A seeded random subset, not the first N. Protocol files are not
            # guaranteed to interleave classes, so a head slice can land almost
            # entirely in one class and silently invalidate the sweep.
            g = torch.Generator().manual_seed(args.seed)
            picks = torch.randperm(len(dataset), generator=g)[:limit].tolist()
            dataset = Subset(dataset, sorted(picks))
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.workers,
        )

    # Report on the held-out split when one is given, else fall back to val.
    report_set = test_set if test_set is not None else val_set
    report_loader = _eval_loader(report_set)
    sweep_loader = _eval_loader(report_set, args.eval_subset)

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adversary=args.adversary,
        gamma=args.gamma,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        attack=AttackConfig(
            epsilon=args.epsilon, alpha=args.alpha,
            num_steps=args.steps, num_restarts=args.restarts,
        ),
        sample_rate=args.sample_rate,
        device=args.device,
        log_every=args.log_every,
    )

    model = RawNet3Detector(sample_rate=args.sample_rate)
    if args.load:
        state = torch.load(args.load, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded weights from {args.load}", flush=True)
    trainer = FSATTrainer(model, config)

    # Evaluation-time attack budget, independent of the training budget.
    eval_attack = AttackConfig(
        epsilon=args.eval_epsilon if args.eval_epsilon is not None else args.epsilon,
        alpha=args.eval_alpha if args.eval_alpha is not None else args.alpha,
        num_steps=args.steps, num_restarts=args.restarts,
    )
    eval_attack_time = AttackConfig(
        epsilon=eval_attack.epsilon,
        alpha=(args.eval_alpha_time if args.eval_alpha_time is not None else eval_attack.alpha),
        num_steps=args.steps, num_restarts=args.restarts,
    )

    print(
        f"training RawNet3{'+RandAug' if args.randaug else ''}"
        f"{'' if args.adversary == 'none' else '+' + args.adversary.upper()} "
        f"on {len(train_set)} samples, device={args.device}",
        flush=True,
    )
    best = {"epoch": None, "eer": float("inf"), "state": None}

    def track_best(epoch: int, metrics: dict) -> None:
        # Lower EER is better. Fall back to average accuracy if a validation
        # batch happened to be single-class and EER came back undefined.
        eer = metrics.get("val_eer")
        score = eer if eer is not None else 1.0 - metrics.get("val_average_accuracy", 0.0)
        if score == score and score < best["eer"]:  # score == score rejects NaN
            best.update(
                epoch=epoch, eer=score,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
            print(f"    new best epoch {epoch} (val_eer={score * 100:.2f}%)", flush=True)

    use_best = args.save_best and val_loader is not None
    if args.save_best and val_loader is None:
        print("--save-best ignored: no --val-manifest given", file=sys.stderr)

    if args.eval_only:
        print("--eval-only: skipping training", flush=True)
        use_best = False
    else:
        trainer.fit(train_loader, val_loader, on_epoch_end=track_best if use_best else None)

    if use_best and best["state"] is not None:
        print(f"\nrestoring best checkpoint from epoch {best['epoch']}", flush=True)
        model.load_state_dict(best["state"])

    report = {"config": vars(args), "history": trainer.history}
    if use_best:
        report["best_epoch"] = best["epoch"]

    report["eval_attack"] = {
        "epsilon": eval_attack.epsilon,
        "alpha_freq": eval_attack.alpha,
        "alpha_time": eval_attack_time.alpha,
        "matches_training_budget": eval_attack.epsilon == args.epsilon,
    }

    # Persist the trained weights BEFORE evaluating. Evaluation is long and can
    # fail on its own (a full-manifest DataLoader is a different beast from the
    # training loop), and losing hours of finished training to a downstream bug
    # is unrecoverable. Saving here makes any eval failure re-runnable.
    if args.save and not args.eval_only:
        torch.save({"model": model.state_dict(), "config": vars(args)}, args.save)
        print(f"\nsaved checkpoint to {args.save}", flush=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({**report, "status": "trained, evaluation pending"}, fh, indent=2, default=str)

    if report_loader is not None:
        split = "test" if test_set is not None else "val"
        clean = trainer.evaluate(report_loader)
        print(f"\nclean ({split}, n={len(report_set)}):  {clean}")
        report["clean"] = clean.as_dict()
        report["report_split"] = split
        report["report_n"] = len(report_set)

        if args.eval_subset and args.eval_subset < len(report_set):
            # State the cap loudly: a truncated sweep otherwise reads as full coverage.
            print(
                f"\nNOTE: attack and corruption sweeps capped at {args.eval_subset} "
                f"of {len(report_set)} utterances.",
                flush=True,
            )
            report["sweep_n"] = args.eval_subset
        else:
            report["sweep_n"] = len(report_set)

        if args.eval_attacks:
            print(f"\nattack budget: eps={eval_attack.epsilon:g} "
                  f"alpha_freq={eval_attack.alpha:g} alpha_time={eval_attack_time.alpha:g}", flush=True)
            domains = evaluate_attack_domains(
                model, sweep_loader, trainer.stft, args.f_lo, args.f_hi, eval_attack, args.device,
                time_config=eval_attack_time,
            )
            print("\nattack domains (Fig. 8a):")
            for name, rep in domains.items():
                print(f"  {name:16s} {rep}")
            report["attack_domains"] = {k: v.as_dict() for k, v in domains.items()}

            bands = evaluate_attack_bands(
                model, sweep_loader, trainer.stft, config=eval_attack, device=args.device
            )
            print("\nattack bands (Table 4):")
            for name, rep in bands.items():
                print(f"  {name:16s} {rep}")
            report["attack_bands"] = {k: v.as_dict() for k, v in bands.items()}

        if args.eval_corruptions:
            corruptions = evaluate_corruptions(
                model, sweep_loader, args.sample_rate, device=args.device, seed=args.seed
            )
            print("\ncorruptions (Fig. 7a):")
            for name, rep in sorted(corruptions.items()):
                print(f"  {name:26s} {rep}")
            report["corruptions"] = {k: v.as_dict() for k, v in corruptions.items()}

    # The checkpoint was already written above, before evaluation. Overwrite the
    # placeholder report with the complete one.
    if args.report:
        report["status"] = "complete"
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nsaved report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
