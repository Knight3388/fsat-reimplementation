"""Assemble the Table 3 comparison from the four run reports.

The paper's headline is not clean accuracy alone: it claims +7.7% on clean and
+29.3% on corrupted-and-attacked audio over the RawNet3 baseline. So this
reports all three regimes side by side.

Usage: python scripts/summarize_runs.py [runs_dir]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

ROWS = [
    ("rawnet3", "RawNet3"),
    ("rawnet3_randaug", "+RandAug"),
    ("rawnet3_randaug_at_time", "+AT(Time)"),
    ("rawnet3_randaug_fsat", "+F-SAT"),
]


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}"


def avg_of(section: dict, key: str = "average_accuracy"):
    """Mean of a metric across every entry in a sweep section."""
    if not section:
        return None
    vals = [v[key] for v in section.values() if v.get(key) is not None]
    return mean(vals) if vals else None


def main(argv=None) -> int:
    runs = Path(argv[0]) if argv else Path(os.environ.get("FSAT_ROOT", ".")) / "runs"
    data = {}
    for key, _ in ROWS:
        path = runs / key / "report.json"
        if path.is_file():
            data[key] = json.load(open(path, encoding="utf-8"))

    if not data:
        print(f"no reports found under {runs}", file=sys.stderr)
        return 1

    any_report = next(iter(data.values()))
    print(f"ASVspoof2019 LA eval, n = {any_report.get('report_n')} "
          f"(attack/corruption sweeps on {any_report.get('sweep_n')} random utterances)\n")

    hdr = f"{'config':<12} {'EER':>7} {'bona':>7} {'spoof':>7} {'clean':>7} {'corrupt':>8} {'attacked':>9} {'epoch':>6}"
    print(hdr)
    print("-" * len(hdr))

    for key, label in ROWS:
        d = data.get(key)
        if d is None:
            print(f"{label:<12} {'(missing)':>7}")
            continue
        c = d["clean"]
        corrupt = avg_of(d.get("corruptions", {}))
        # Attacked = mean average-accuracy across every band and domain probe,
        # excluding the untouched 'clean' reference the sweep includes.
        att = {k: v for k, v in {**d.get("attack_domains", {}), **d.get("attack_bands", {})}.items()
               if "clean" not in k.lower()}
        attacked = avg_of(att)
        print(f"{label:<12} {pct(c['eer']):>7} {pct(c['real_accuracy']):>7} "
              f"{pct(c['fake_accuracy']):>7} {pct(c['average_accuracy']):>7} "
              f"{pct(corrupt):>8} {pct(attacked):>9} {str(d.get('best_epoch','?')):>6}")

    base = data.get("rawnet3")
    fsat = data.get("rawnet3_randaug_fsat")
    at = data.get("rawnet3_randaug_at_time")
    if base and fsat:
        print("\nF-SAT vs RawNet3 baseline (paper claims +7.7 clean, +29.3 corrupted/attacked):")
        d_clean = (fsat["clean"]["average_accuracy"] - base["clean"]["average_accuracy"]) * 100
        bc, fc = avg_of(base.get("corruptions", {})), avg_of(fsat.get("corruptions", {}))
        print(f"  clean avg-accuracy   {d_clean:+.2f} pp")
        if bc is not None and fc is not None:
            print(f"  corrupted avg-acc    {(fc - bc) * 100:+.2f} pp")
        print(f"  EER                  {(fsat['clean']['eer'] - base['clean']['eer']) * 100:+.2f} pp")
    if at and fsat:
        print("\nF-SAT vs isotropic AT(Time) — the paper's actual novelty claim:")
        print(f"  EER                  {(fsat['clean']['eer'] - at['clean']['eer']) * 100:+.2f} pp")
        print(f"  clean avg-accuracy   {(fsat['clean']['average_accuracy'] - at['clean']['average_accuracy']) * 100:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
