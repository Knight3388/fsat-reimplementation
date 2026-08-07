"""Paired min t-DCF and per-attack EER across seeds, for both arms.

Consumes the per-utterance dumps written by `fsat-train --dump-scores` and
reports what the aggregate reports cannot: the ASVspoof2019 primary metric, and
which attacks the difference between the two arms actually comes from.

Usage:
    python scripts/tdcf_summary.py <dump_root> --asv-scores <file> --meta <file>

where <dump_root> holds `<arm>[_s<seed>]/scores.tsv`.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_analysis as sa  # noqa: E402

ARMS = [("rawnet3_randaug_at_time", "AT(Time)"), ("rawnet3_randaug_fsat", "F-SAT")]
SEEDS = [1000, 1001, 1002, 1003, 1004]


def analyse(path: Path, asv, meta):
    scores = sa.read_scores(str(path))
    if not scores:
        return None
    bona = np.array([s for k, s in scores.values() if k == "bonafide"])
    spoof = np.array([s for k, s in scores.values() if k != "bonafide"])
    eer, _ = sa.compute_eer(bona, spoof)

    tar, non, sp = asv
    _, asv_thr = sa.compute_eer(tar, non)
    Pfa, Pmiss, Pmiss_spoof = sa.obtain_asv_error_rates(tar, non, sp, asv_thr)
    tdcf, _, _, _ = sa.compute_min_tdcf(bona, spoof, Pfa, Pmiss, Pmiss_spoof)

    per_attack = {}
    if meta:
        buckets = {}
        for uid, (key, s) in scores.items():
            if key != "bonafide":
                buckets.setdefault(meta.get(uid, "?"), []).append(s)
        for atk, arr in buckets.items():
            per_attack[atk] = sa.compute_eer(bona, np.array(arr))[0] * 100
    return {"eer": eer * 100, "tdcf": tdcf, "per_attack": per_attack}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--asv-scores", required=True)
    p.add_argument("--meta")
    args = p.parse_args(argv)

    root = Path(args.root)
    asv = sa.read_asv(args.asv_scores)
    meta = sa.read_meta(args.meta) if args.meta else {}

    res = {}
    for seed in SEEDS:
        for name, lab in ARMS:
            d = root / (name if seed == 1000 else f"{name}_s{seed}") / "scores.tsv"
            if d.is_file():
                r = analyse(d, asv, meta)
                if r:
                    res[(seed, lab)] = r

    have = [s for s in SEEDS if (s, "AT(Time)") in res and (s, "F-SAT") in res]
    print(f"complete pairs: {len(have)}  {have}\n")

    print(f"{'seed':<6}{'arm':<11}{'EER %':>9}{'min t-DCF':>12}")
    print("-" * 38)
    for seed in have:
        for _, lab in ARMS:
            r = res[(seed, lab)]
            print(f"{seed:<6}{lab:<11}{r['eer']:>9.2f}{r['tdcf']:>12.5f}")

    for metric in ("eer", "tdcf"):
        diffs = [res[(s, "F-SAT")][metric] - res[(s, "AT(Time)")][metric] for s in have]
        mean = st.mean(diffs)
        print(f"\npaired {metric} (F-SAT minus AT(Time)); lower is better for both:")
        print("  per seed: " + ", ".join(f"{d:+.4f}" for d in diffs))
        if len(diffs) > 1:
            sd = st.stdev(diffs)
            lo, hi = sa_bootstrap(diffs)
            print(f"  mean {mean:+.4f}  sd {sd:.4f}")
            print(f"  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'excludes 0' if (lo > 0) == (hi > 0) else 'INCLUDES 0'}")
        wins = sum(1 for d in diffs if d < 0)
        print(f"  F-SAT better in {wins}/{len(diffs)} seeds")

    attacks = sorted({a for s in have for a in res[(s, 'AT(Time)')]["per_attack"]})
    if attacks:
        print(f"\nper-attack EER %, mean over {len(have)} seeds:")
        print(f"  {'attack':<8}{'AT(Time)':>10}{'F-SAT':>10}{'diff':>9}")
        print("  " + "-" * 37)
        for atk in attacks:
            a = st.mean([res[(s, "AT(Time)")]["per_attack"].get(atk, float("nan")) for s in have])
            f = st.mean([res[(s, "F-SAT")]["per_attack"].get(atk, float("nan")) for s in have])
            flag = "  <-- dominant" if max(a, f) > 5 else ""
            print(f"  {atk:<8}{a:>10.2f}{f:>10.2f}{f - a:>+9.2f}{flag}")
    return 0


def sa_bootstrap(diffs, iters=20000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(st.mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


if __name__ == "__main__":
    raise SystemExit(main())
