"""Build F-SAT manifests from the ASVspoof2019 LA distribution.

The paper (arXiv:2411.00121v1, Table 2) reports ASVspoof2019 as a public
evaluation corpus, so it is the practical target for reproduction. This turns
the official CM protocol files into the ``path<TAB>label`` manifests that
:class:`fsat.data.ManifestDataset` reads.

Protocol lines look like::

    LA_0079 LA_T_1138215 - -   bonafide
    LA_0079 LA_T_1271820 - A01 spoof

Column 2 is the utterance id and column 5 is the key. Columns 3-4 (the
attack/system id) are kept in a sidecar ``.meta.tsv`` so per-attack breakdowns
stay possible later without re-parsing the protocol.

Usage::

    python scripts/make_asvspoof19_manifests.py \
        --la-root "/home/sameera/unofficail_audio_deepfake/Dataset/LA" \
        --out-dir manifests/asvspoof19
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# (split name, protocol filename, audio subdirectory)
SPLITS: List[Tuple[str, str, str]] = [
    ("train", "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    ("dev", "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    ("eval", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
]

# Official counts, used as an integrity check rather than a hard requirement.
EXPECTED: Dict[str, int] = {"train": 25380, "dev": 24844, "eval": 71237}

VALID_KEYS = {"bonafide", "spoof"}


def parse_protocol(path: Path) -> List[Tuple[str, str, str]]:
    """Return ``(utt_id, key, system_id)`` triples from a CM protocol file."""
    rows: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{lineno}: expected >=5 fields, got {line!r}")
            utt_id, system_id, key = parts[1], parts[3], parts[4].lower()
            if key not in VALID_KEYS:
                raise ValueError(f"{path}:{lineno}: unexpected key {key!r}")
            rows.append((utt_id, key, system_id))
    return rows


def build_split(
    la_root: Path, out_dir: Path, split: str, protocol_name: str, audio_subdir: str,
    check_exists: bool,
) -> Dict[str, object]:
    protocol = la_root / "ASVspoof2019_LA_cm_protocols" / protocol_name
    flac_dir = la_root / audio_subdir / "flac"

    if not protocol.is_file():
        raise FileNotFoundError(f"protocol not found: {protocol}")
    if not flac_dir.is_dir():
        raise FileNotFoundError(f"audio directory not found: {flac_dir}")

    rows = parse_protocol(protocol)

    manifest_path = out_dir / f"{split}.tsv"
    meta_path = out_dir / f"{split}.meta.tsv"
    missing: List[str] = []
    n_real = n_fake = 0

    with open(manifest_path, "w", encoding="utf-8", newline="\n") as mf, \
         open(meta_path, "w", encoding="utf-8", newline="\n") as sf:
        sf.write("#utt_id\tsystem_id\tkey\n")
        for utt_id, key, system_id in rows:
            audio = flac_dir / f"{utt_id}.flac"
            # Checking every file costs one stat() per utterance. Worth it once,
            # because a missing file only surfaces mid-epoch otherwise.
            if check_exists and not audio.is_file():
                missing.append(utt_id)
                continue
            label = "bonafide" if key == "bonafide" else "spoof"
            mf.write(f"{audio}\t{label}\n")
            sf.write(f"{utt_id}\t{system_id}\t{key}\n")
            if key == "bonafide":
                n_real += 1
            else:
                n_fake += 1

    written = n_real + n_fake
    expected = EXPECTED.get(split)
    status = "ok"
    if expected is not None and len(rows) != expected:
        status = f"protocol has {len(rows)} lines, official count is {expected}"
    if missing:
        status = f"{len(missing)} audio files missing (first: {missing[0]})"

    return {
        "split": split,
        "protocol_lines": len(rows),
        "written": written,
        "real": n_real,
        "fake": n_fake,
        "ratio": (n_fake / n_real) if n_real else float("inf"),
        "missing": len(missing),
        "status": status,
        "manifest": str(manifest_path),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--la-root", required=True,
        help="ASVspoof2019 LA root (contains ASVspoof2019_LA_cm_protocols/ and ASVspoof2019_LA_*/).",
    )
    p.add_argument("--out-dir", required=True, help="Directory to write the manifests into.")
    p.add_argument(
        "--no-check-exists", action="store_true",
        help="Skip the per-file existence check (faster, but defers failures to training).",
    )
    args = p.parse_args(argv)

    la_root = Path(args.la_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for split, protocol_name, audio_subdir in SPLITS:
        report = build_split(
            la_root, out_dir, split, protocol_name, audio_subdir,
            check_exists=not args.no_check_exists,
        )
        reports.append(report)
        print(
            f"{report['split']:6s} wrote {report['written']:>6d}  "
            f"real={report['real']:>6d}  fake={report['fake']:>6d}  "
            f"fake:real={report['ratio']:.2f}:1  [{report['status']}]",
            flush=True,
        )

    problems = [r for r in reports if r["status"] != "ok"]
    if problems:
        print(
            "\nWARNING: "
            + "; ".join(f"{r['split']}: {r['status']}" for r in problems)
            + "\nDo not report numbers from a split flagged above without resolving it first.",
            file=sys.stderr,
        )

    train = next(r for r in reports if r["split"] == "train")
    print(
        f"\nTrain set is imbalanced at {train['ratio']:.1f}:1 fake:real. "
        "Pass --balanced-sampler to fsat-train, or the real class collapses.",
    )
    print(f"manifests in {os.path.abspath(out_dir)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
