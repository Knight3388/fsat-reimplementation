"""Paired seed analysis for the F-SAT vs isotropic-AT comparison.

Both arms of each seed share initialisation and data order, so the per-seed
difference is paired and the shared run-to-run variance cancels. That matters a
lot here: raw EER swings over a point between seeds, while the paired difference
is far tighter.

Reports mean, sd, a paired t-statistic and a bootstrap CI on the difference,
plus the seed count that would be needed to resolve the observed effect.

Usage: python scripts/paired_analysis.py [runs_dir] [--seeds 1000 1001 ...]
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
from pathlib import Path

ARMS = [("rawnet3_randaug_at_time", "AT(Time)"), ("rawnet3_randaug_fsat", "F-SAT")]
METRICS = ["eer", "clean_avg", "attacked", "corrupted"]


def load(runs: Path, name: str, seed: int):
    d = runs / (name if seed == 1000 else f"{name}_s{seed}")
    p = d / "report.json"
    if not p.is_file():
        return None
    j = json.load(open(p, encoding="utf-8"))
    if j.get("status") != "complete":
        return None
    att = [v["average_accuracy"] for s in ("attack_domains", "attack_bands")
           for n, v in j.get(s, {}).items() if "no_attack" not in n]
    cor = [v["average_accuracy"] for v in j.get("corruptions", {}).values()]
    return {
        "eer": j["clean"]["eer"] * 100,
        "clean_avg": j["clean"]["average_accuracy"] * 100,
        "attacked": st.mean(att) * 100 if att else float("nan"),
        "corrupted": st.mean(cor) * 100 if cor else float("nan"),
    }


def bootstrap_ci(diffs, iters=20000, alpha=0.05, seed=0):
    """Percentile bootstrap on the mean paired difference."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(st.mean([diffs[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return lo, hi


def main(argv):
    runs = Path(argv[0]) if argv and not argv[0].startswith("-") else Path(
        os.environ.get("FSAT_ROOT", ".")) / "runs"
    seeds = [1000, 1001, 1002, 1003, 1004]

    data = {}
    for seed in seeds:
        for name, lab in ARMS:
            r = load(runs, name, seed)
            if r:
                data[(seed, lab)] = r

    have = [s for s in seeds if (s, "AT(Time)") in data and (s, "F-SAT") in data]
    print(f"complete pairs: {len(have)} of {len(seeds)}  ({have})\n")

    print(f"{'seed':<6}{'arm':<11}" + "".join(f"{m:>11}" for m in METRICS))
    print("-" * (17 + 11 * len(METRICS)))
    for seed in have:
        for _, lab in ARMS:
            r = data[(seed, lab)]
            print(f"{seed:<6}{lab:<11}" + "".join(f"{r[m]:>11.2f}" for m in METRICS))

    print("\n" + "=" * 62)
    print("PAIRED DIFFERENCE  (F-SAT minus AT(Time), per seed)")
    print("=" * 62)
    for m in METRICS:
        diffs = [data[(s, "F-SAT")][m] - data[(s, "AT(Time)")][m] for s in have]
        mean = st.mean(diffs)
        sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
        n = len(diffs)
        se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")
        t = mean / se if se == se and se > 0 else float("nan")
        lo, hi = bootstrap_ci(diffs) if n > 1 else (float("nan"),) * 2
        better = "F-SAT better" if (mean < 0) == (m == "eer") else "AT(Time) better"
        # For EER lower is better; for the accuracy metrics higher is better.
        wins = sum(1 for d in diffs if (d < 0) == (m == "eer"))
        print(f"\n{m}:")
        print("  per seed: " + ", ".join(f"{d:+.2f}" for d in diffs))
        if n > 1:
            print(f"  mean {mean:+.3f}  sd {sd:.3f}  t({n-1}) = {t:+.2f}")
            # nan comparisons silently evaluate False, which made a single-seed
            # run print "excludes 0" for a CI that does not exist.
            verdict = "excludes 0" if (lo > 0) == (hi > 0) else "INCLUDES 0"
            print(f"  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        else:
            print(f"  mean {mean:+.3f}   (n=1: no sd, no CI, no significance)")
        print(f"  {better} in {wins}/{n} seeds")
        if sd == sd and sd > 0 and abs(mean) > 0:
            need = 7.85 * (sd / abs(mean)) ** 2  # paired, 80% power, alpha=.05
            print(f"  seeds needed for 80% power at this effect: {max(3, math.ceil(need))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
