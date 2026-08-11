"""One consolidated table of every configuration, against the F-SAT baseline.

The reference row is the paper's own method as specified (F-SAT, 4-8 kHz,
gamma 0.1, pretrained init), so every delta reads as "what this change buys or
costs relative to what the paper proposes".

Deltas are computed PAIRED where seeds overlap — the same seed on both sides,
which matters because seed effects here are large and common-mode (one seed
gives ~3.1% EER across every configuration while others give ~1.5-2.0%).

Usage: python scripts/master_table.py [--root <fsat root>] [--sibling <FSAT-Exps>]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

SEEDS = [1000, 1001, 1002, 1003, 1004]
BASELINE = "F-SAT (paper)"

# (label, tree, directory-stem, group)
CONFIGS = [
    ("RawNet3",            "paper",  "rawnet3",                   "Table 3"),
    ("+RandAug",           "paper",  "rawnet3_randaug",           "Table 3"),
    ("+AT(Time)",          "paper",  "rawnet3_randaug_at_time",   "Table 3"),
    (BASELINE,             "paper",  "rawnet3_randaug_fsat",      "Table 3"),
    ("band 0-8 kHz",       "tuning", "band_0_8k",                 "Band"),
    ("band 2-8 kHz",       "tuning", "band_2_8k",                 "Band"),
    ("band 6-8 kHz",       "tuning", "band_6_8k",                 "Band"),
    ("gamma 0.3",          "tuning", "gamma_0.3",                 "Gamma"),
    ("gamma 1.0",          "tuning", "gamma_1.0",                 "Gamma"),
    ("gamma 2.0",          "tuning", "gamma_2.0",                 "Gamma"),
]


def load(path: Path):
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
        "clean": d["clean"]["average_accuracy"] * 100,
        "attacked": st.mean(att) * 100 if att else float("nan"),
        "corrupted": st.mean(cor) * 100 if cor else float("nan"),
    }


def series(root: Path, sibling: Path | None, tree: str, stem: str):
    """Per-seed results for one configuration, keyed by seed."""
    out = {}
    for s in SEEDS:
        if tree == "paper":
            name = stem if s == 1000 else f"{stem}_s{s}"
            r = load(root / "runs_paper" / name / "report.json")
        else:
            r = load(root / "runs_tuning" / f"{stem}_s{s}" / "report.json")
            if r is None and sibling is not None:
                # Seed 1000 of each tuning variant was run first in the sibling tree.
                for sweep in ("band_sweep", "gamma_sweep"):
                    r = r or load(sibling / "runs" / sweep / stem / "report.json")
        if r:
            out[s] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("FSAT_ROOT", "."))
    p.add_argument("--sibling", default=None)
    a = p.parse_args()
    root, sib = Path(a.root), (Path(a.sibling) if a.sibling else None)

    data = {label: series(root, sib, tree, stem) for label, tree, stem, _ in CONFIGS}
    base = data[BASELINE]

    metrics = [("eer", "EER", True), ("clean", "clean", False),
               ("attacked", "attacked", False), ("corrupted", "corrupt", False)]

    hdr = f"{'configuration':<16}{'n':>3}" + "".join(f"{h:>19}" for _, h, _ in metrics)
    print("All values mean over seeds. Delta is PAIRED vs the F-SAT baseline row.")
    print("EER: lower better. clean/attacked/corrupt: higher better.\n")
    print(hdr)
    print("-" * len(hdr))

    group = None
    for label, _, _, grp in CONFIGS:
        d = data[label]
        if not d:
            print(f"{label:<16}{'--':>3}   (no complete runs)")
            continue
        if grp != group:
            group = grp
            print(f"[{grp}]")
        line = f"{label:<16}{len(d):>3}"
        for key, _, lower_better in metrics:
            mean = st.mean(v[key] for v in d.values())
            shared = sorted(set(d) & set(base))
            if label == BASELINE or not shared:
                line += f"{mean:>12.2f}{'':>7}"
            else:
                diff = st.mean(d[s][key] - base[s][key] for s in shared)
                good = (diff < 0) if lower_better else (diff > 0)
                mark = "+" if good else ("-" if abs(diff) > 1e-9 else " ")
                line += f"{mean:>12.2f}{diff:>+6.2f}{mark}"
        print(line)

    print("\n'+' = better than the F-SAT baseline, '-' = worse.")
    print(f"Baseline row = {BASELINE}: 4-8 kHz band, gamma 0.1, pretrained init,")
    print("i.e. the configuration the paper itself specifies.")


if __name__ == "__main__":
    main()
