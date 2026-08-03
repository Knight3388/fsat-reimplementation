"""Tests for manifest parsing.

Only parsing is covered here: reading actual audio needs ``soundfile`` and real
files, so those tests are skipped when the optional extra is absent.
"""

import pytest

from fsat.data import ManifestDataset
from fsat.metrics import FAKE, REAL

def _has_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401

        return True
    except ImportError:
        return False


def write(tmp_path, text, name="manifest.tsv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parses_tab_separated_entries(tmp_path):
    manifest = write(tmp_path, "a.wav\treal\nb.wav\tfake\n")
    ds = ManifestDataset(manifest)
    assert ds.items == [("a.wav", REAL), ("b.wav", FAKE)]


def test_parses_space_separated_entries(tmp_path):
    manifest = write(tmp_path, "some dir/a.wav bonafide\nother/b.wav spoof\n")
    ds = ManifestDataset(manifest)
    assert ds.items == [("some dir/a.wav", REAL), ("other/b.wav", FAKE)]


@pytest.mark.parametrize(
    "label,expected",
    [("0", REAL), ("real", REAL), ("bonafide", REAL), ("genuine", REAL),
     ("1", FAKE), ("fake", FAKE), ("spoof", FAKE), ("deepfake", FAKE),
     ("REAL", REAL), ("Spoof", FAKE)],
)
def test_label_aliases(tmp_path, label, expected):
    ds = ManifestDataset(write(tmp_path, f"a.wav\t{label}\n"))
    assert ds.items[0][1] == expected


def test_comments_and_blank_lines_ignored(tmp_path):
    manifest = write(tmp_path, "# header\n\na.wav\treal\n\n# trailing\nb.wav\tfake\n")
    assert len(ManifestDataset(manifest)) == 2


def test_unknown_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown label"):
        ManifestDataset(write(tmp_path, "a.wav\tmaybe\n"))


def test_malformed_line_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        ManifestDataset(write(tmp_path, "just_a_path_no_label\n"))


def test_empty_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match="no usable entries"):
        ManifestDataset(write(tmp_path, "# only a comment\n"))


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ManifestDataset(str(tmp_path / "nope.tsv"))


@pytest.mark.skipif(_has_soundfile(), reason="soundfile is installed")
def test_reading_audio_without_soundfile_gives_a_clear_error(tmp_path):
    ds = ManifestDataset(write(tmp_path, "a.wav\treal\n"))
    with pytest.raises(ImportError, match="soundfile"):
        ds[0]


@pytest.mark.skipif(not _has_soundfile(), reason="needs soundfile")
def test_reads_and_crops_real_audio(tmp_path):
    import numpy as np
    import soundfile as sf
    import torch

    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(16000 * 5, dtype=np.float32), 16000)
    ds = ManifestDataset(write(tmp_path, f"{path}\treal\n"), duration=2.0)
    x, label = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (32000,) and label == REAL
