# Replication results, and how we got them wrong twice first

This documents an attempted replication of F-SAT (Zhang et al., ICLR 2025,
arXiv:2411.00121) on **ASVspoof2019 LA**, and — deliberately — the errors made
along the way. If you are a student reading this to learn how reproduction
works, the mistakes section is the more useful half.

**Scope.** The paper's headline is measured on DeepFakeVox-HQ, a 1.3 M-sample
corpus that was never released and cannot be rebuilt (5 of its 14 deepfake
sources are paid commercial APIs, and the real half of its test set is scraped
from YouTube). Nothing here contradicts the paper's own numbers, which are
uncheckable by anyone outside its authors' lab. What is tested is whether the
*claim* transfers to the standard public benchmark.

---

## Setup

| | |
|---|---|
| Corpus | ASVspoof2019 LA — train 25,380 / dev 24,844 / **eval 71,237** |
| Model | RawNet3, 16.3 M params, reimplemented from arXiv:2203.08488 |
| Selection | best dev EER, reported on the held-out eval split |
| Seeds | 5, **paired** (both arms share a seed) |
| Unstated in the paper | optimizer, LR, schedule, batch size, epochs, clip length — all ours (Adam + cosine, 1e-3, batch 32, 30 epochs, 4 s crops) |

Table 3's four rows, single seed:

| Config | EER | Bonafide | Spoof | Clean avg |
|---|---|---|---|---|
| RawNet3 | 10.82 | 99.90 | 69.71 | 84.80 |
| +RandAug | 9.33 | 99.52 | 75.95 | 87.74 |
| +AT(Time) | 8.08 | 98.52 | 86.69 | 92.61 |
| +F-SAT | 7.71 | 97.34 | 88.42 | 92.88 |

**This table is superseded** — see *Mistake 1*. It is kept because it is what
the flawed run produced.

---

## What replicated

The clean-accuracy gain over the RawNet3 baseline: **+8.07 pp** against the
paper's claimed **+7.7**. Close, and reassuring given that six training
hyperparameters had to be guessed.

## What did not

The paper's actual contribution is that confining adversarial perturbation to
4–8 kHz beats perturbing the whole waveform. On ASVspoof2019 it does not.

The paper quotes ε=0.01 (frequency) and ε=1e-4 (time) but never says whether
1e-4 is the *time-domain* budget or the *evaluation* budget for everything.
Rather than pick one, all three defensible readings were run — 5 paired seeds
each, on identical checkpoints:

| Evaluation budget | Attacked gap | 95% CI | Seeds F-SAT wins |
|---|---|---|---|
| ε=1e-4 everywhere | −2.58 | [−3.10, −2.03] | 0/5 |
| Domain-matched (freq 0.01, time 1e-4) | −5.63 | [−7.66, −3.59] | 0/5 |
| ε=0.01 everywhere | −6.37 | [−8.38, −4.28] | 0/5 |

Negative favours isotropic AT. The gap is **largest** under the reading that
tests F-SAT at its own training budget, which is where it should be strongest.

### min t-DCF, the metric ASVspoof2019 is judged on

EER alone is not comparable to this corpus's literature.

| | AT(Time) | F-SAT |
|---|---|---|
| mean min t-DCF | **0.1510** | 0.1749 |
| mean EER % | **6.94** | 7.78 |

Paired difference **+0.0240** t-DCF, CI [+0.0134, +0.0339], **0/5 seeds**
favour F-SAT. For calibration, the official baselines are CQCC-GMM 0.2366 and
LFCC-GMM 0.2116, so this system is respectable but far from modern SOTA.

### Per attack

F-SAT is worse on **11 of 13** attacks. A17 and A18 carry nearly all the
absolute error, and F-SAT is worse on both — it does not trade average
performance for robustness on the hard cases.

| Attack | AT(Time) | F-SAT | Diff |
|---|---|---|---|
| A18 | 23.32 | 26.78 | **+3.46** |
| A10 | 1.71 | 3.98 | +2.27 |
| A14 | 0.80 | 2.79 | +1.99 |
| A15 | 1.04 | 2.89 | +1.85 |
| A17 | 15.67 | 16.69 | +1.02 |
| others | < 1.8 | < 3 | |

---

## Four mistakes, and how each was caught

### Mistake 1 — the baseline was crippled, so the method "won"

The first run applied `--epsilon 0.01 --alpha 0.004` to *every* configuration.
The paper specifies **different budgets per domain**: ε=0.01/α=4e-4 in
frequency, ε=1e-4/α=4e-5 in time. So the isotropic-AT baseline was trained with
a time-domain perturbation **100× too large**, and F-SAT's own step size was
**10× too large**.

Reported at the time: F-SAT ahead by **+23.4 pp**. After correcting both arms:
F-SAT **behind**. The entire result was an artifact of a handicapped baseline.

**Lesson.** When your method beats a baseline by a suspiciously large margin,
audit the baseline before celebrating. A weak baseline is the most common way a
true-looking result is manufactured, and it is easy to do to yourself by
accident.

### Mistake 2 — evaluating at the training budget

The attack sweep initially ran at ε=0.01, the *training* epsilon. F-SAT trains
on exactly that perturbation in exactly the 4–8 kHz band being probed, so it
was being tested on its training distribution. Re-evaluating at the paper's
ε=1e-4 shrank the gap from +23.4 to +3.3 pp; fixing Mistake 1 then flipped the
sign.

**Lesson.** Evaluation conditions that coincide with training conditions
flatter whichever method trained on them. When a paper is ambiguous about
which to use, run every reading rather than picking one.

### Mistake 3 — reporting identical numbers as a real result

The job watcher read `runs/<name>/report.json` while a multi-seed sweep wrote
to `runs/<name>_s<seed>/`. It therefore reported **seed 1000's numbers for
every seed**. Seed 1001 was announced with figures identical to seed 1000 to
the last digit.

Caught only because identical-to-the-digit results across different random
seeds are impossible. Had the seeds differed slightly, this would have silently
produced fake agreement.

**Lesson.** Results that look too clean are a bug report, not a finding.

### Mistake 4 — losing 75 minutes of finished training

`--save` ran *after* evaluation. A fully trained model (30 epochs, val EER
0.08%) was discarded when evaluation crashed on file-descriptor exhaustion over
the 71,237-utterance eval set.

**Lesson.** Persist expensive artifacts the moment they exist, before anything
that can fail independently. Checkpoints now save before evaluation.

### Two smaller ones

- `--eval-subset` originally took the **first** N utterances. The eval manifest
  has runs of up to 96 consecutive `spoof` entries, so a head slice was badly
  class-skewed. Now a seeded random sample.
- The analysis script printed `excludes 0` for a single-seed confidence
  interval, because NaN comparisons silently evaluate false. It now refuses to
  report significance at n=1.

---

## Statistics

**Pairing did the heavy lifting.** Both arms share a seed, so initialisation
and data order are identical and shared variance cancels:

| | Spread |
|---|---|
| Raw EER between seeds, same arm | ~1.2 pp |
| Paired difference | sd 0.6–0.7 |

Unpaired, resolving a ~3 pp effect would need roughly 15–20 seeds per arm.
Paired, **3 suffice**. Five were run.

**Honest limits.**

- With n=5, a Wilcoxon signed-rank test **cannot reach p<0.05** — its minimum
  two-sided p is 0.0625. The reported t-statistics and bootstrap CIs assume
  approximate normality of the paired differences, reasonable but unverifiable
  at this size. A distribution-free result needs ≥6 seeds.
- The `RawNet3` and `+RandAug` rows are **single-seed**.
- t-DCF here is an independent implementation of the ASVspoof2019 evaluation
  plan, not the organisers' script. It is known-answer checked (perfect
  detector → 0.0, useless detector → 1.0), the ASV operating point reproduces
  the published ≈2.48% EER at 2.46%, pooled EER matches the training reports
  exactly through a separate code path, and A17/A18 emerge as hardest, which is
  the documented ASVspoof2019 pattern.
- **Seeds fix variance, not bias.** Every error above was systematic. A hundred
  seeds would have reproduced them faithfully and returned a tight confidence
  interval around a wrong number.

---

## Verdict

The implementation replicates. The central claim does not transfer to
ASVspoof2019 under any reading of the paper's attack budget.

This is **not** a refutation. The claim is made on data nobody else can obtain,
and it may well hold there. What can be said is narrower and still worth
saying: on the standard public benchmark, with the baseline trained at the
budget the paper specifies for it, band-selective adversarial training does not
beat plain isotropic adversarial training.

Reproduce with `scripts/paired_analysis.py`, `scripts/score_analysis.py` and
`scripts/tdcf_summary.py`.
