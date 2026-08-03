"""Tests for the command-line entrypoint and its helpers."""

import json
from collections import Counter

import pytest
import torch

from fsat.cli import _balanced_sampler, _dataset_labels, build_parser, main
from fsat.data import ManifestDataset, SyntheticSpeechDataset
from fsat.metrics import FAKE, REAL


# ------------------------------------------------------------------ parser
def test_defaults_follow_the_paper():
    args = build_parser().parse_args([])
    assert args.adversary == "fsat"
    assert args.gamma == 0.1            # Fig. 9b
    assert (args.f_lo, args.f_hi) == (4000.0, 8000.0)  # Table 4
    assert args.epsilon == 0.01         # Fig. 9a


def test_rejects_unknown_adversary():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--adversary", "nonsense"])


# ------------------------------------------------------------------ labels
def test_reads_labels_from_synthetic_dataset():
    ds = SyntheticSpeechDataset(num_items=10, duration=1.0)
    labels = _dataset_labels(ds)
    assert len(labels) == 10
    assert Counter(labels) == {REAL: 5, FAKE: 5}


def test_reads_labels_from_manifest_without_decoding_audio(tmp_path):
    manifest = tmp_path / "m.tsv"
    manifest.write_text("a.wav\treal\nb.wav\tfake\nc.wav\tfake\n", encoding="utf-8")
    # These paths do not exist; reading labels must not touch the filesystem.
    assert _dataset_labels(ManifestDataset(str(manifest))) == [REAL, FAKE, FAKE]


def test_unknown_dataset_type_rejected():
    with pytest.raises(TypeError):
        _dataset_labels(object())


# ------------------------------------------------------------------ sampler
def test_balanced_sampler_equalizes_a_skewed_dataset(tmp_path):
    """Weights must be inversely proportional to class frequency."""
    manifest = tmp_path / "skewed.tsv"
    manifest.write_text("r.wav\treal\n" + "f.wav\tfake\n" * 9, encoding="utf-8")
    sampler = _balanced_sampler(ManifestDataset(str(manifest)))

    weights = torch.as_tensor(sampler.weights, dtype=torch.double)
    assert len(weights) == 10
    assert float(weights[0]) == pytest.approx(1.0)        # 1 / 1 real
    assert float(weights[1]) == pytest.approx(1.0 / 9)    # 1 / 9 fakes
    # Total probability mass per class must match.
    assert float(weights[0]) == pytest.approx(float(weights[1:].sum()))


def test_balanced_sampler_draws_both_classes(tmp_path):
    manifest = tmp_path / "skewed.tsv"
    manifest.write_text("r.wav\treal\n" + "f.wav\tfake\n" * 99, encoding="utf-8")
    ds = ManifestDataset(str(manifest))
    sampler = _balanced_sampler(ds)

    torch.manual_seed(0)
    labels = _dataset_labels(ds)
    drawn = Counter(labels[i] for i in list(sampler))
    # Without balancing this would be 1:99. Allow generous slack for sampling noise.
    assert drawn[REAL] > len(labels) * 0.3


# ------------------------------------------------------------------ end to end
def test_main_runs_and_writes_a_report(tmp_path):
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "model.pt"
    exit_code = main([
        "--adversary", "fsat", "--randaug", "--balanced-sampler",
        "--epochs", "1", "--train-items", "8", "--val-items", "4",
        "--batch-size", "4", "--duration", "1.0", "--steps", "1",
        "--log-every", "0",
        "--report", str(report), "--save", str(checkpoint),
    ])
    assert exit_code == 0
    assert checkpoint.exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(payload["history"]) == 1
    assert "clean" in payload
    assert payload["config"]["gamma"] == 0.1


def test_save_best_records_the_chosen_epoch(tmp_path):
    report = tmp_path / "report.json"
    main([
        "--adversary", "none", "--save-best",
        "--epochs", "2", "--train-items", "8", "--val-items", "4",
        "--batch-size", "4", "--duration", "1.0", "--log-every", "0",
        "--report", str(report),
    ])
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["best_epoch"] in (1, 2)
