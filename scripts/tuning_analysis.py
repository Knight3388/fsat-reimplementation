"""Paired analysis of the band-placement and gamma tuning sweeps.

Each variant is compared against the paper's own choice at the SAME seed, so
the comparison is paired and the (large) seed effect cancels. That matters here:
seed 1002 produces ~3.1% EER across every band, while other seeds sit near
1.5-2.0%, so an unpaired comparison would be swamped by which seeds landed
where.

Baselines, both from the paper-faithful sweep:
  band  4000-8000 Hz  -> runs_paper/rawnet3_randaug_fsat[_s<seed>]
  gamma 0.1           -> the same runs (gamma 0.1 is the authors' value)

Usage: python scripts/tuning_analysis.py [--root <fsat root>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
from pathlib import Path

SEEDS = [1000, 1001, 1002, 1003, 1004]
METRICS = ["eer", "clean_avg", "attacked", "corrupted"]
# Lower is better for EER; higher is better for the accuracy metrics.
LOWER_BETTER = {"eer"}


def summarise(path: Path):
    if not path.is_file():
        return None
    d = json.load(open(path, encoding="utf-8"))
    if d.get("status") != "complete":
        return None
    att = [v["average_accuracy"] for s in ("attack_domains", "attack_bands")
           for n, v in d.get(s, {}).items() if "no_attack" not in n]
    cor = [v["average_accuracy"] for v in d.get("corruptions", {}).values()]
    return {
        "eer": d["clean"]["eer"] * 100,
        "clean_avg": d["clean"]["average_accuracy"] * 100,
        "attacked": st.mean(att) * 100 if att else float("nan"),
        "corrupted": st.mean(cor) * 100 if cor else float("nan"),
    }


def baseline(root: Path, seed: int):
    name = "rawnet3_randaug_fsat" if seed == 1000 else f"rawnet3_randaug_fsat_s{seed}"
    return summarise(root / "runs_paper" / name / "report.json")


def variant(root: Path, tag: str, seed: int, sibling: Path | None):
    r = summarise(root / "runs_tuning" / f"{tag}_s{seed}" / "report.json")
    if r or sibling is None:
        return r
    # Seed 1000 of each variant was run first in the sibling ablation tree.
    for sweep in ("band_sweep", "gamma_sweep"):
        r = summarise(sibling / "runs" / sweep / tag / "report.json")
        if r:
            return r
    return None


def bootstrap_ci(diffs, iters=20000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(st.mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def report(label: str, rows):
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for metric in METRICS:
        diffs = [v[metric] - b[metric] for b, v in rows]
        if not diffs:
            continue
        mean = st.mean(diffs)
        n = len(diffs)
        better = sum(1 for d in diffs
                     if (d < 0) == (metric in LOWER_BETTER))
        line = f"  {metric:<11} mean {mean:+7.3f}"
        if n > 1:
            sd = st.stdev(diffs)
            lo, hi = bootstrap_ci(diffs)
            sig = "excludes 0" if (lo > 0) == (hi > 0) else "includes 0"
            line += f"  sd {sd:5.3f}  CI [{lo:+7.3f},{hi:+7.3f}] {sig}"
            if sd > 0 and mean != 0:
                need = max(3, math.ceil(7.85 * (sd / abs(mean)) ** 2))
                line += f"  n80={need}"
        line += f"  better {better}/{n}"
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("FSAT_ROOT", "."))
    p.add_argument("--sibling", default=None,
                   help="FSAT-Exps tree holding the seed-1000 variants.")
    a = p.parse_args()
    root = Path(a.root)
    sib = Path(a.sibling) if a.sibling else None

    print("Each variant paired against the paper's own choice at the same seed.")
    print("EER: lower is better. clean_avg / attacked / corrupted: higher is better.")

    for tag, label in [
        ("band_0_8k", "BAND 0-8 kHz  vs  paper's 4-8 kHz"),
        ("band_2_8k", "BAND 2-8 kHz  vs  paper's 4-8 kHz"),
        ("band_6_8k", "BAND 6-8 kHz  vs  paper's 4-8 kHz"),
        ("gamma_0.3", "GAMMA 0.3  vs  paper's 0.1"),
        ("gamma_1.0", "GAMMA 1.0  vs  paper's 0.1"),
    ]:
        rows = []
        for s in SEEDS:
            b, v = baseline(root, s), variant(root, tag, s, sib)
            if b and v:
                rows.append((b, v))
        if rows:
            report(f"{label}   ({len(rows)} paired seeds)", rows)
        else:
            print(f"\n{label}: no complete pairs found")


if __name__ == "__main__":
    main()
