"""Datasets and batching.

Two dataset types are provided:

* :class:`SyntheticSpeechDataset` -- procedurally generated audio requiring no
  corpus. Fake clips carry a faint *high-frequency* artifact, which mirrors the
  paper's central empirical finding (Fig. 2): detectors latch onto
  high-frequency content that humans cannot hear. This makes the smoke test a
  real end-to-end check (the model must actually learn) and lets the
  band-placement behaviour of F-SAT be exercised without downloading anything.

* :class:`ManifestDataset` -- reads a plain-text manifest of ``path<TAB>label``
  for real corpora (DeepFakeVox-HQ, ASVspoof2019, WaveFake). Requires
  ``soundfile``.

Both yield fixed-length crops so batches are rectangular, and both optionally
apply audio RandAugment (Section 4.2) on the training split only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import FAKE, REAL
from .randaugment import RANDAUGMENT_OPS, rand_augment


@dataclass
class AugmentConfig:
    """RandAugment settings (Section 4.2).

    ``n`` is the number of transforms drawn per sample and ``p`` the
    probability that augmentation is applied at all.
    """

    enabled: bool = False
    n: int = 2
    p: float = 0.5
    pool: Optional[Sequence[str]] = None
    per_op_probability: bool = False


def _crop_or_pad(x: np.ndarray, length: int, rng: np.random.Generator, random_crop: bool) -> np.ndarray:
    if x.size == length:
        return x
    if x.size > length:
        start = int(rng.integers(0, x.size - length + 1)) if random_crop else 0
        return x[start : start + length]
    reps = int(np.ceil(length / max(x.size, 1)))
    return np.tile(x, reps)[:length]


class _BaseAudioDataset(Dataset):
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        augment: Optional[AugmentConfig] = None,
        random_crop: bool = True,
        seed: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_samples = int(round(sample_rate * duration))
        self.augment = augment or AugmentConfig()
        self.random_crop = random_crop
        self.seed = seed

    def _rng(self, index: int) -> np.random.Generator:
        # Per-item seeding keeps DataLoader workers reproducible and independent.
        return np.random.default_rng((self.seed * 1_000_003 + index) % (2**32))

    def _finalize(self, x: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        if self.augment.enabled:
            x = rand_augment(
                x,
                sr=self.sample_rate,
                n=self.augment.n,
                p=self.augment.p,
                rng=rng,
                pool=self.augment.pool or RANDAUGMENT_OPS,
                per_op_probability=self.augment.per_op_probability,
            )
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


class SyntheticSpeechDataset(_BaseAudioDataset):
    """Procedural real/fake audio with a high-frequency artifact in the fakes.

    Real clips are a voiced-speech caricature: a wandering fundamental with a
    natural harmonic roll-off, an amplitude envelope, and a breath-noise floor.
    Fake clips are the same generator plus two artifacts of the kind neural
    vocoders leave behind, both confined above ``artifact_floor_hz``:

    * a faint fixed-frequency tone pair (spectral "spikes"), and
    * a band-limited noise excess in the top octave.

    ``artifact_strength`` controls how audible the give-away is. The default is
    deliberately small so the task is non-trivial but learnable in a few
    hundred steps on CPU.
    """

    def __init__(
        self,
        num_items: int = 256,
        sample_rate: int = 16000,
        duration: float = 3.0,
        augment: Optional[AugmentConfig] = None,
        artifact_strength: float = 0.02,
        artifact_floor_hz: float = 5000.0,
        seed: int = 0,
        random_crop: bool = False,
    ) -> None:
        super().__init__(sample_rate, duration, augment, random_crop, seed)
        self.num_items = num_items
        self.artifact_strength = artifact_strength
        self.artifact_floor_hz = artifact_floor_hz
        # Balanced, deterministic label assignment.
        self.labels = [REAL if i % 2 == 0 else FAKE for i in range(num_items)]

    def __len__(self) -> int:
        return self.num_items

    def _synth_voice(self, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(self.num_samples, dtype=np.float64) / self.sample_rate
        f0 = rng.uniform(90.0, 220.0)
        # Slow pitch contour, as in connected speech.
        contour = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t + rng.uniform(0, 6.28)))
        phase = 2 * np.pi * np.cumsum(contour) / self.sample_rate

        signal = np.zeros_like(t)
        n_harmonics = int(self.sample_rate / 2 / f0)
        for k in range(1, min(n_harmonics, 60) + 1):
            # -12 dB/octave roll-off, the usual glottal source slope.
            signal += (1.0 / k**2) * np.sin(k * phase + rng.uniform(0, 6.28))

        # Syllabic amplitude envelope.
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t + rng.uniform(0, 6.28))
        signal *= envelope
        signal += 0.005 * rng.normal(size=t.size)  # breath noise
        return signal / (np.max(np.abs(signal)) + 1e-9)

    def _add_artifacts(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(self.num_samples, dtype=np.float64) / self.sample_rate
        nyquist = self.sample_rate / 2.0
        out = x.copy()

        # Fixed-frequency spectral spikes above the artifact floor.
        for _ in range(2):
            f = rng.uniform(self.artifact_floor_hz, nyquist * 0.98)
            out += self.artifact_strength * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))

        # Excess energy in the top octave, shaped in the frequency domain.
        spectrum = np.fft.rfft(rng.normal(size=self.num_samples))
        freqs = np.fft.rfftfreq(self.num_samples, d=1.0 / self.sample_rate)
        spectrum[freqs < self.artifact_floor_hz] = 0.0
        band_noise = np.fft.irfft(spectrum, n=self.num_samples)
        band_noise /= np.max(np.abs(band_noise)) + 1e-9
        out += self.artifact_strength * 1.5 * band_noise

        return out / (np.max(np.abs(out)) + 1e-9)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rng = self._rng(index)
        label = self.labels[index]
        x = self._synth_voice(rng)
        if label == FAKE:
            x = self._add_artifacts(x, rng)
        return self._finalize(x.astype(np.float32), rng), label


class ManifestDataset(_BaseAudioDataset):
    """Audio files listed in a ``path<TAB>label`` manifest.

    ``label`` is ``0``/``real``/``bonafide`` or ``1``/``fake``/``spoof``
    (case-insensitive). Lines starting with ``#`` and blank lines are ignored.
    Requires ``soundfile`` (``pip install 'fsat[audio]'``).
    """

    _LABEL_MAP = {
        "0": REAL, "real": REAL, "bonafide": REAL, "genuine": REAL,
        "1": FAKE, "fake": FAKE, "spoof": FAKE, "deepfake": FAKE,
    }

    def __init__(
        self,
        manifest: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        augment: Optional[AugmentConfig] = None,
        random_crop: bool = True,
        seed: int = 0,
        root: Optional[str] = None,
    ) -> None:
        super().__init__(sample_rate, duration, augment, random_crop, seed)
        self.root = root
        self.items: List[Tuple[str, int]] = []
        with open(manifest, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
                if len(parts) != 2:
                    raise ValueError(f"{manifest}:{lineno}: expected 'path<TAB>label', got {line!r}")
                path, label = parts[0].strip(), parts[1].strip().lower()
                if label not in self._LABEL_MAP:
                    raise ValueError(f"{manifest}:{lineno}: unknown label {label!r}")
                self.items.append((path, self._LABEL_MAP[label]))
        if not self.items:
            raise ValueError(f"{manifest} contains no usable entries")

    @property
    def labels(self) -> List[int]:
        """Label of every entry, in manifest order."""
        return [label for _, label in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "ManifestDataset needs soundfile. Install with: uv pip install soundfile"
            ) from exc

        path, label = self.items[index]
        if self.root:
            path = os.path.join(self.root, path)
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            from .randaugment import _resample  # local import keeps the module lean

            audio = _resample(audio, self.sample_rate / sr)

        rng = self._rng(index)
        audio = _crop_or_pad(audio, self.num_samples, rng, self.random_crop)
        return self._finalize(audio.astype(np.float32), rng), label


def collate(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack ``(waveform, label)`` pairs into ``((B, T), (B,))``."""
    waveforms = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


__all__ = [
    "AugmentConfig",
    "SyntheticSpeechDataset",
    "ManifestDataset",
    "collate",
]
