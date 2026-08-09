# Replication results, and how we got them wrong three times first

> [!IMPORTANT]
> **Final verdict, and it is a null.** The authors' code turned out to exist, as
> an ICLR supplementary archive linked from nowhere. Re-run with *their*
> hyperparameters over 5 paired seeds, **F-SAT and plain isotropic time-domain
> adversarial training are statistically indistinguishable on ASVspoof2019** —
> every metric's confidence interval includes zero. Resolving the residual
> differences would need **266-298 paired seeds**.
>
> | Metric | F-SAT − AT(Time) | 95% CI | Seeds needed |
> |---|---|---|---|
> | EER | −0.046 | [−0.253, +0.183] | 266 |
> | Clean avg | +0.151 | [−1.178, +1.584] | 1133 |
> | Attacked | −0.458 | [−2.150, +2.117] | 298 |
> | Corrupted | +0.632 | [−0.627, +2.047] | — |
>
> The one large effect is not F-SAT at all: **fine-tuning a pretrained RawNet3
> cut EER roughly fourfold, ~7.5% to ~1.8%.** That single choice dwarfs
> everything the paper's method contributes on this corpus.
>
> **Two follow-up sweeps at 5 paired seeds each:** the paper's 4-8 kHz band
> choice **holds up** (no alternative beats it; only the narrow 6-8 kHz is
> clearly worse), but its γ=0.1 does not — **γ=1.0 buys +8.81 pp of adversarial
> robustness at no measurable cost**, in 5/5 seeds.
>
> Everything below the *Setup* section describes the earlier, misconfigured
> attempts. They are kept deliberately — the sequence of being confidently wrong
> three times is the most instructive part of this repository.



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

## Tuning sweeps: is the paper's γ and band choice right?

Each variant paired against the paper's own choice at the same seed, 5 seeds.
Pairing is essential here: seed 1002 produces ~3.1% EER across *every* band while
other seeds sit near 1.5-2.0%, so unpaired comparisons are swamped by seed noise.

### Band placement — the paper's 4-8 kHz holds up

| vs 4-8 kHz | EER | Attacked | Verdict |
|---|---|---|---|
| 0-8 kHz | +0.280 [+0.001,+0.604] | +2.56 (includes 0) | slightly worse EER |
| 2-8 kHz | +0.167 (includes 0) | +0.90 (includes 0) | **indistinguishable** |
| 6-8 kHz | +0.457 [+0.268,+0.744] | **−5.79 [−7.10,−4.48]** | **worse on all 4 metrics, 0/5 seeds** |

A single-seed sweep had suggested 2-8 kHz beat 4-8 kHz on every metric. **It does
not replicate.** At 5 paired seeds the two are indistinguishable, and the
apparent advantage was seed noise — the exact error this document warns about
elsewhere, committed again.

What does survive: **6-8 kHz is decisively worse**, losing on all four metrics in
all five seeds, most heavily under attack (−5.79 pp). So bandwidth matters and
too narrow a band hurts, but the paper's specific 4-8 kHz selection is not
beaten by any alternative tested.

### γ — the paper's 0.1 leaves real robustness unclaimed

| vs γ=0.1 | EER | Attacked | Seeds better |
|---|---|---|---|
| γ=0.3 | +0.071 (includes 0) | **+3.34 [+1.31,+5.03]** | 4/5 |
| γ=1.0 | −0.013 (includes 0) | **+8.81 [+7.19,+10.46]** | **5/5** |

**γ=1.0 buys +8.81 pp of adversarial robustness at no measurable cost to EER,
clean accuracy or corruption robustness.** Every seed agrees, the interval sits
far from zero, and only 3 seeds are needed for 80% power. The paper's Fig. 9b
selects γ=0.1.

This is the one place where a specific tuning choice in the paper is clearly
improvable, and it is worth more than the method's headline claim: γ moves
robustness by 8.8 points while band-selective-versus-isotropic moves it by 0.5.

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

### Mistake 5 — the code existed the whole time

We reimplemented from the paper text after concluding no code was released,
having checked Hugging Face, the GitHub API under several term sets, the arXiv
full text including appendix, and the ICLR poster page. All negative.

The authors' full training and evaluation pipeline was sitting in a 10 MB
**supplementary archive on the ICLR proceedings page**, linked from nowhere and
surfaced by no search. It settles every hyperparameter the paper omits, and we
had guessed ten of them wrong — including that they *fine-tune a pretrained
RawNet3 at lr=1e-5*, where we trained from scratch at 1e-3.

Two whole result sets had to be withdrawn because of it.

**Lesson.** Before concluding "no code released", check the conference
supplementary archive. NeurIPS, ICLR and ICML all host them, papers frequently
fail to mention them, and search engines do not index them.

### Mistake 6 — quoting a power estimate from four points

At four paired seeds, F-SAT led on EER in 4/4 with a small standard deviation,
and the power calculation said ~9 seeds would settle it. We reported that and
queued five more runs.

The fifth pair came in against F-SAT, tripled the standard deviation, and the
required sample size jumped from 9 to **266**. The confidence interval, which had
excluded zero, swallowed it.

**Lesson.** A standard deviation from four observations is barely an estimate,
and every quantity derived from it inherits that. The queued extension was
cancelled once the arithmetic was honest, because five more seeds cannot resolve
an effect that needs 266.

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

The implementation replicates. Run with the authors' own configuration, the
central claim is **not reproduced and not refuted** — on ASVspoof2019 the two
methods are indistinguishable, and the difference is too small to resolve with
any feasible number of seeds.

This is **not** a refutation of the paper. Its claim is measured on
DeepFakeVox-HQ, which nobody else can obtain, and it may well hold there. What
can be said is narrower: on the standard public benchmark, band-selective
adversarial training confers no measurable advantage over plain isotropic
adversarial training, while the choice of pretrained initialisation dominates
both.

If you take one thing from this document, take the fourfold EER improvement from
initialisation. It is a much larger effect than the method under study, it is
mentioned nowhere in the paper, and it was only discoverable by reading code the
paper does not cite.

Reproduce with `scripts/paired_analysis.py`, `scripts/score_analysis.py` and
`scripts/tdcf_summary.py`.
