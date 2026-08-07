"""min t-DCF and per-attack EER from dumped per-utterance scores.

ASVspoof2019's primary metric is minimum normalised t-DCF, not EER, so results
quoted only as EER are not comparable to the published literature on this
corpus. This computes t-DCF following the ASVspoof2019 evaluation plan
(Kinnunen et al., Odyssey 2018) and breaks EER down by attack id.

The countermeasure scores come from `fsat-train --dump-scores`, which writes
`cm_score = log P(bonafide) - log P(spoof)` (higher = more bonafide, the
ASVspoof convention).

Usage:
    python scripts/score_analysis.py scores.tsv \
        --asv-scores /path/to/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
        --meta manifests/asvspoof19/eval.meta.tsv

NOTE: this is an independent implementation of the official scoring, not the
organisers' script. The cost model is printed with every result so the
parameters can be checked.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np

# ASVspoof2019 LA evaluation plan, Table 1.
COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,   # 0.9405
    "Pnon": (1 - 0.05) * 0.01,   # 0.0095
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """FRR/FAR over every threshold. Higher score = more 'target'."""
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    idx = np.nanargmin(np.abs(frr - far))
    return float((frr[idx] + far[idx]) / 2), float(thresholds[idx])


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, threshold):
    Pfa_asv = float(np.sum(non_asv >= threshold) / non_asv.size)
    Pmiss_asv = float(np.sum(tar_asv < threshold) / tar_asv.size)
    Pmiss_spoof_asv = float(np.sum(spoof_asv < threshold) / spoof_asv.size)
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_min_tdcf(bonafide_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv):
    """Minimum normalised t-DCF, ASVspoof2019 evaluation plan eq. (5)-(8)."""
    c = COST_MODEL
    C1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * Pmiss_asv) - \
        c["Pnon"] * c["Cfa_asv"] * Pfa_asv
    C2 = c["Cfa_cm"] * c["Pspoof"] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        raise ValueError(
            f"negative t-DCF cost coefficient (C1={C1:.4g}, C2={C2:.4g}); "
            "the ASV operating point or cost model is inconsistent"
        )

    Pmiss_cm, Pfa_cm, thresholds = compute_det_curve(bonafide_cm, spoof_cm)
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / min(C1, C2)
    i = int(np.argmin(tDCF_norm))
    return float(tDCF_norm[i]), float(thresholds[i]), C1, C2


def read_scores(path: str) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("utt_id"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            out[parts[0]] = (parts[1], float(parts[4]))
    return out


def read_meta(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def read_asv(path: str):
    tar, non, spoof = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            kind, score = parts[1], float(parts[2])
            if kind == "target":
                tar.append(score)
            elif kind == "nontarget":
                non.append(score)
            elif kind == "spoof":
                spoof.append(score)
    return np.array(tar), np.array(non), np.array(spoof)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scores", help="TSV from --dump-scores")
    p.add_argument("--asv-scores", help="ASVspoof2019 ASV scores file (enables t-DCF)")
    p.add_argument("--meta", help="eval.meta.tsv sidecar (enables per-attack EER)")
    args = p.parse_args(argv)

    scores = read_scores(args.scores)
    if not scores:
        print(f"no usable rows in {args.scores}", file=sys.stderr)
        return 1

    bona = np.array([s for k, s in scores.values() if k == "bonafide"])
    spoof = np.array([s for k, s in scores.values() if k != "bonafide"])
    print(f"utterances: {len(scores)}  bonafide={bona.size}  spoof={spoof.size}")

    eer, thr = compute_eer(bona, spoof)
    print(f"\npooled EER: {eer * 100:.2f}%  (threshold {thr:+.4f})")

    if args.asv_scores:
        tar_asv, non_asv, spoof_asv = read_asv(args.asv_scores)
        asv_eer, asv_thr = compute_eer(tar_asv, non_asv)
        Pfa, Pmiss, Pmiss_spoof = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_thr)
        tdcf, tdcf_thr, C1, C2 = compute_min_tdcf(bona, spoof, Pfa, Pmiss, Pmiss_spoof)
        print(f"\nASV operating point (its own EER): {asv_eer * 100:.2f}%  thr {asv_thr:+.4f}")
        print(f"  Pfa_asv={Pfa:.4f}  Pmiss_asv={Pmiss:.4f}  Pmiss_spoof_asv={Pmiss_spoof:.4f}")
        print(f"  cost coefficients C1={C1:.5f} C2={C2:.5f}")
        print(f"\nmin t-DCF: {tdcf:.5f}  (CM threshold {tdcf_thr:+.4f})")
        print(f"  cost model: {COST_MODEL}")

    if args.meta:
        meta = read_meta(args.meta)
        by_attack: Dict[str, List[float]] = {}
        for uid, (key, s) in scores.items():
            if key == "bonafide":
                continue
            by_attack.setdefault(meta.get(uid, "?"), []).append(s)
        print("\nper-attack EER (each attack scored against ALL bonafide):")
        rows = []
        for atk in sorted(by_attack):
            arr = np.array(by_attack[atk])
            e, _ = compute_eer(bona, arr)
            rows.append((atk, arr.size, e * 100))
        for atk, n, e in rows:
            print(f"  {atk:<6} n={n:<6} EER={e:6.2f}%")
        if rows:
            worst = max(rows, key=lambda r: r[2])
            best = min(rows, key=lambda r: r[2])
            print(f"  worst: {worst[0]} ({worst[2]:.2f}%)   best: {best[0]} ({best[2]:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
