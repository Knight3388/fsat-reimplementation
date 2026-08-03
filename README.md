# F-SAT — Frequency-Selective Adversarial Training

Reimplementation of:

> Zirui Zhang, Wei Hao, Aroon Sankoh, William Lin, Emanuel Mendiola-Ortiz,
> Junfeng Yang, Chengzhi Mao.
> **"I Can Hear You: Selective Robust Training for Deepfake Audio Detection."**
> arXiv:2411.00121v1 (ICLR 2025).

Self-contained and dependency-light: PyTorch, NumPy and SciPy only. No
`asteroid-filterbanks`, no `audiomentations`, no dataset download required to
run or test it.

---

## What the paper proposes

Three separable contributions. This repo implements the two that are methods
(the third is a dataset, which is not redistributable here):

| Contribution | Status |
|---|---|
| **F-SAT** — adversarial training on STFT *magnitude*, confined to a high-frequency band | Implemented |
| **RandAugment for audio** — 22 randomized corruptions applied during training | Implemented (all 22) |
| **DeepFakeVox-HQ** — 1.3 M-sample dataset, 270 k high-quality deepfakes | Not included (see [Using real data](#using-real-data)) |

The motivating finding (Fig. 2) is that detectors rely on **high-frequency
content inaudible to humans**, so an attacker who perturbs only that region
breaks them. F-SAT trains against exactly that perturbation instead of an
isotropic one, which is why it hardens the model without the usual clean
accuracy collapse of standard adversarial training.

---

## Install

```bash
uv sync
```

Optional, only needed for `ManifestDataset` (reading real audio files):

```bash
uv pip install soundfile
```

## Verify the install

```bash
uv run pytest -q
```

165 tests, ~25 s on CPU. They check the STFT operator against the paper's
formulas, band confinement of the perturbation, PGD monotonicity, the sinc
filterbank's analytic structure, and the `L_total = L_clean + γ·L_robust`
identity.

## End-to-end demo

```bash
uv run python scripts/smoke_test.py
```

Trains three of the Table 3 configurations on procedurally generated audio and
compares them clean, corrupted and attacked. ~13 min on CPU. Observed:

```
Approach                   Clean  Corrupt  Atk(t)  Atk(f)  Atk(ph)
RawNet3                    96.9%    64.4%    4.7%   90.6%    96.9%
RawNet3+RandAug            87.5%    68.8%    1.6%   75.0%    87.5%
RawNet3+RandAug+F-SAT      84.4%    71.2%   12.5%   81.2%    84.4%
```

The qualitative pattern the paper reports does appear: RandAug buys corruption
robustness, F-SAT adds robustness to both attack domains over its matched
control, and phase attacks are by far the weakest (which is the paper's stated
reason for perturbing magnitude). **These are not the paper's numbers** — this
is a small synthetic task, so differences of a few points are within noise.

---

## Quick start

```python
import torch
from torch.utils.data import DataLoader
from fsat import (RawNet3Detector, FSATTrainer, TrainConfig, AttackConfig,
                  SyntheticSpeechDataset, AugmentConfig, collate)

train = SyntheticSpeechDataset(256, augment=AugmentConfig(enabled=True, n=2, p=0.5))
loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=collate)

trainer = FSATTrainer(
    RawNet3Detector(),
    TrainConfig(
        adversary="fsat",
        gamma=0.1,                                    # Fig. 9b
        f_lo=4000.0, f_hi=8000.0,                     # Table 4
        attack=AttackConfig(epsilon=0.01, num_steps=5),  # Fig. 9a
    ),
)
trainer.fit(loader)
```

Or from the command line:

```bash
uv run fsat-train --adversary fsat --randaug --epochs 20 --eval-attacks
```

---

## How the method maps to the code

### The band-selective operator, `s(x, δ, f_l, f_u)`

`src/fsat/stft.py`. Given the STFT `X` with magnitude `X_ρ` and phase `X_φ`:

```
r_l = floor(f_l · n_fft / sr)          r_u = ceil(f_u · n_fft / sr)
D_kk = 1 if r_l ≤ k ≤ r_u else 0
δ_s  = D · δ
X'   = (X_ρ + δ_s) · e^{j·X_φ}
x'   = ISTFT(X')
```

`D` is diagonal, so it is stored as its diagonal and applied elementwise.
Phase is carried as a unit complex phasor `X / |X|` rather than an angle —
identical to `e^{j·X_φ}`, but without the `atan2` gradient singularity at
`|X| = 0`.

### The F-SAT objective

`src/fsat/trainer.py`:

```
L_robust = H(F_θ(x'), y)
L_clean  = H(F_θ(x),  y)
L_total  = L_clean + γ · L_robust
```

The inner maximization runs `K` PGD steps (`src/fsat/attacks.py`), projecting
onto the ε-ball after each step.

### RandAugment for audio

`src/fsat/randaugment.py` implements all 22 transforms of Fig. 10, plus four
more (`gain`, `polarity_inversion`, `limiter`, `clipping_distortion`) needed to
complete the 24-corruption evaluation battery of Fig. 7a.

```python
from fsat import rand_augment, RANDAUGMENT_OPS, CORRUPTION_SUITE
augmented = rand_augment(waveform, sr=16000, n=2, p=0.5)
```

Every transform is length-preserving so batches stay rectangular, which means
`time_stretch` stretches and then refits to the original duration.

### RawNet3

`src/fsat/models/rawnet3.py`, a from-scratch port of Jung et al.
(arXiv:2203.08488) — 16.3 M parameters:

- `ParamSincFB`: analytic parameterized sinc filterbank, mel-spaced at init,
  learning only each filter's low cutoff and bandwidth. Emits `n/2` cosine
  kernels followed by their `n/2` sine partners.
- `Bottle2neck` ×3: Res2Net blocks with dilation 2/3/4, residual paths and
  attentive feature-map scaling.
- Multi-layer feature aggregation, then attentive statistics pooling with
  context.

---

## Reproducing the paper's ablations

```python
from fsat.evaluation import (evaluate_attack_bands, evaluate_attack_domains,
                             evaluate_corruptions)

evaluate_attack_domains(model, loader, stft)   # Fig. 8a / Table 3: time vs magnitude vs phase
evaluate_attack_bands(model, loader, stft)     # Table 4 / Table 5: 0-8k, 2-8k, 4-8k, 6-8k
evaluate_corruptions(model, loader)            # Fig. 7a: the 24-corruption battery
```

Table 3's four rows correspond to:

| Table 3 row | Configuration |
|---|---|
| `RawNet3` | `--adversary none` |
| `RawNet3+RandAug` | `--adversary none --randaug` |
| `RawNet3+RandAug+AT(Time)` | `--adversary time --randaug` |
| `RawNet3+RandAug+F-SAT` | `--adversary fsat --randaug` |

Transfer (black-box) attacks — Table 5's source columns `A` / `A'` / `B` — use
`trainer.evaluate_under_attack_from(loader, attack, source_model)`.

### ⚠ Two different epsilons

The paper uses **ε = 0.01 for training** (Fig. 9a) but **ε = 1e-4 for
evaluation** (Table 5, with α = 4e-4 in frequency and α = 4e-5 in time). These
differ by 100×. Attacking at the training epsilon drives every model to near
0% accuracy and makes the comparison meaningless. The defaults here follow the
paper: `TrainConfig.attack` defaults to ε = 0.01, and the smoke test evaluates
at ε = 1e-4.

### ⚠ ε on STFT magnitude is scale-dependent

The paper gives ε as a bare number on the STFT magnitude but never states its
STFT normalization or `n_fft`. Unnormalized magnitudes scale with both, so the
same ε is a large perturbation on a quiet clip and a negligible one on a loud
one. **Hold `n_fft` fixed when comparing runs.**

`AttackConfig.epsilon_mode` exposes the choice:

- `"absolute"` (default) — the paper's literal reading.
- `"relative"` — rescales ε and α per utterance by the mean in-band magnitude,
  giving gain invariance. Useful on corpora with inconsistent levels.

Measured spread of time-domain perturbation RMS across `n_fft ∈ {512, 1024,
2048}`: ~2.1× absolute, ~3.2× relative. Relative mode fixes gain sensitivity,
**not** `n_fft` sensitivity.

---

## Selected hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Band `[f_l, f_u]` | 4–8 kHz | Table 4, best of four bands |
| `γ` (robust/clean loss ratio) | 0.1 | Fig. 9b |
| `ε` (training) | 0.01 | Fig. 9a |
| `ε`, `α` (evaluation, frequency) | 1e-4, 4e-4 | Table 5 |
| `ε`, `α` (evaluation, time) | 1e-4, 4e-5 | Table 5 |
| PGD iterations | 2 or 5 | Table 5 |
| PGD restarts | 1 or 2 | Table 5 |
| RandAugment `N`, `p` | not stated | defaults `N=2, p=0.5` here |
| Optimizer, LR, batch size, epochs | not stated | Adam + cosine, configurable |

---

## Where this departs from the paper, and why

The paper leaves several things unspecified or ambiguous. Each choice below is
flagged in the source at the point it is made.

1. **PGD updates `δ`, not `x'`.** Eq. (6) is written as an update on `x'` but
   applies the frequency mask `M` and takes the gradient sign on the right.
   Fig. 4 shows the loop updating the magnitude-domain perturbation `δ`. We
   follow Fig. 4 — the only reading under which both `M` and the projection
   onto `‖x' − x‖ ≤ ε` are well defined. Restricted to the band, the two are
   equivalent.

2. **Magnitude is clamped at zero.** Eq. (5) does not bound `X_ρ + δ_s`. A
   sufficiently negative `δ` makes it negative, which is not a magnitude but a
   180° phase flip smuggled through the magnitude channel. Clamping keeps the
   perturbation inside the subspace the method claims to act on. Disable with
   `clamp_magnitude=False`.

3. **Fig. 10's `p` gates the whole chain, not each transform.** The reference
   snippet evaluates `np.random.random() < p` once for the entire chain, and
   `np.random.choice` samples *with* replacement. Both behaviours are
   reproduced by default; `per_op_probability=True` and `replace=False` switch
   them off.

4. **Attacks are generated in eval mode.** Standard adversarial-training
   practice, so the inner maximization does not corrupt BatchNorm running
   statistics. Clean and adversarial batches are forwarded separately rather
   than concatenated, to avoid mixing their batch statistics.

5. **`background_noise` falls back to synthetic noise.** The paper mixes real
   recordings. Pass `noise_pool=` to use a corpus.

6. **Optimizer and schedule are ours.** Not stated in the paper.

7. **`pitch_shift` and `time_stretch` use a phase vocoder.** The specific
   implementation is not stated.

8. **STFT settings are ours, and ε depends on them.** The paper states neither
   `n_fft`, hop length, nor whether the STFT is normalized, so its ε values are
   not directly portable. Defaults here are `n_fft=1024, hop=256`, unnormalized.
   See the ε-scale warning above.

---

## Using real data

Write a `path<TAB>label` manifest, where label is `0`/`real`/`bonafide` or
`1`/`fake`/`spoof`:

```
/data/asvspoof19/LA_E_1234567.flac	bonafide
/data/wavefake/melgan/LJ001-0001.wav	spoof
```

For ASVspoof2019 LA, `scripts/make_asvspoof19_manifests.py` builds all three
splits from the official CM protocol files:

```bash
uv run python scripts/make_asvspoof19_manifests.py \
  --la-root /path/to/LA --out-dir manifests/asvspoof19
```

It verifies every referenced `.flac` exists, checks the split sizes against the
official counts (25,380 / 24,844 / 71,237), reports the class imbalance, and
writes a `.meta.tsv` sidecar with the attack/system id per utterance so
per-attack breakdowns stay possible later.

```bash
uv run fsat-train \
  --train-manifest train.tsv --val-manifest test.tsv \
  --randaug --adversary fsat \
  --balanced-sampler --save-best \
  --epochs 50 --batch-size 32 --device cuda \
  --eval-attacks --eval-corruptions --report results.json
```

Audio is resampled to `--sample-rate` and randomly cropped to `--duration`
seconds. `ManifestDataset` requires `soundfile`.

Two flags matter on real corpora:

- `--balanced-sampler` — anti-spoofing sets are heavily skewed (ASVspoof2019 LA
  train is 2,580 bonafide against 22,800 spoof). Trained unweighted, a detector
  reaches ~90% pooled accuracy by answering "spoof" everywhere while real-class
  accuracy goes to zero. Since the paper reports real, fake and average
  accuracy separately, that failure is visible in the metrics — this flag
  prevents it.
- `--save-best` — keeps the lowest-val-EER epoch rather than the last one.
  Adversarial training is noisy epoch to epoch, so the final checkpoint is
  often not the best one.

**On DeepFakeVox-HQ**: the paper's own training corpus (690 k real + 640 k fake
utterances) is not bundled here. The paper's Table 2 also reports ASVspoof2019
and WaveFake, both publicly available, which are the practical targets for
reproduction with this code.

---

## Scope

This is a faithful implementation of the *method*, verified for correctness on
synthetic audio. It is **not** a numerical reproduction of the paper's tables —
that needs DeepFakeVox-HQ, the full-size model, and a real training schedule on
GPU. The smoke test checks that the mechanism behaves as described (the
detector learns a high-frequency cue, a band-limited attack destroys it), not
that any specific accuracy is matched.

The attack and training core is deliberately model-agnostic: `FSATTrainer` and
the attacks accept any callable mapping a waveform `(B, T)` to logits `(B, C)`,
so the same F-SAT objective can be applied to a detector other than RawNet3.

---

## Layout

```
src/fsat/
  stft.py           BandSelectiveSTFT — the s(x, δ, f_l, f_u) operator
  attacks.py        FrequencySelective / TimeDomain / Phase PGD attacks
  trainer.py        FSATTrainer — the min-max objective
  randaugment.py    22 RandAugment transforms + the 24-corruption battery
  evaluation.py     Band, domain and corruption sweeps
  data.py           Synthetic and manifest-backed datasets
  metrics.py        Real/fake/average accuracy and EER
  models/rawnet3.py RawNet3 and the detector head
  cli.py            fsat-train entrypoint
tests/              165 tests
scripts/
  smoke_test.py               3-way comparison on synthetic audio
  make_asvspoof19_manifests.py  ASVspoof2019 LA manifest builder
```
