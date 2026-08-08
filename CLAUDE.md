# F-SAT reimplementation — instructions for Claude

Read this before touching anything in this repo. It exists because two people and
two Claude sessions work on this project, and per-user Claude memory is **not**
shared between them. Everything durable lives here.

**What this is:** an unofficial reimplementation of F-SAT (Zhang et al., ICLR
2025, arXiv:2411.00121), evaluated on ASVspoof2019 LA. See
[RESULTS.md](RESULTS.md) for the outcome and [README.md](README.md) for the
method.

---

## The single most important fact

**The authors DID release code.** It is an ICLR 2025 supplementary `.zip` on
`proceedings.iclr.cc`, linked from nowhere — not the arXiv page, not the poster
page, no GitHub, and no web search surfaces it. Before ever concluding "no code
was released" for this or any paper, **check the conference supplementary
archive**. NeurIPS, ICLR and ICML all host them.

It settles every hyperparameter the paper omits. We guessed ten of them wrong and
had to withdraw two complete result sets.

### The authors' actual configuration

| Parameter | Value | Source |
|---|---|---|
| Init | **fine-tunes pretrained RawNet3** | `train_attack_fake_frequency.py:35` |
| lr / epochs | **1e-5** / 10 | `hyperparameters.py` |
| Schedule | warmup 1 epoch → cosine, warmup_lr 1e-6, min_lr 1e-7 | `lr_schedule.py` |
| Batch / gamma | 32 / 0.1 | `hyperparameters.py` |
| STFT | n_fft 1024, **hop 512**, Hann, **no magnitude clamp** | `frequency_attack.py` |
| Frequency attack | eps **0.005**, alpha **0.002**, 2 iters, 1 restart | `hyperparameters.py` |
| Time attack | eps **5e-4**, alpha **1e-4**, 2 iters | `hyperparameters.py` |
| **Evaluation** attack | eps 5e-4, alpha 1e-4, **5 restarts x 10 iters** | `eval_pgd.py` |
| RandAugment | **n=1, p=1.0** | `aug_num`, `aug_prob` |
| Band | 4000 Hz to Nyquist | `spectrum_target_freq=4000` |

`sbatch/fsat_as19_paper.sbatch` encodes all of this. **Use it.** Do not
reconstruct these values from the paper prose — that is how we got it wrong.

There is **no epsilon ambiguity**: frequency and time are separate parameters in
their code. Any analysis treating "eps=0.01 vs 1e-4" as two readings of one
number is solving a non-problem.

---

## Traps that have already cost real time

**The pretrained prefix.** This model nests the backbone one level deeper than
the public checkpoint, so **0 of 234 tensors match by name**. With
`strict=False` — which the authors' code uses — that loads nothing, silently, and
trains from random init while appearing to fine-tune. `load_pretrained()` in
`cli.py` prefixes keys with `backbone.` (226/234 match) and **aborts** below
`--min-pretrained-frac`. Never remove that guard.

**Class imbalance.** ASVspoof2019 LA train is 8.8:1 spoof:bonafide. Unweighted,
the model reaches ~90% pooled accuracy by answering "spoof" always while
bonafide accuracy goes to zero. Always pass `--balanced-sampler`. A third-party
replication failed on exactly this.

**Eval-set ordering.** The eval manifest has runs of up to 96 consecutive
`spoof` entries. Any head-slice subsetting is badly class-skewed —
`--eval-subset` takes a seeded random sample for this reason.

**Save before evaluating.** `--save` writes the checkpoint *before* the eval
sweeps, because a fully trained model was once lost to a file-descriptor crash
during evaluation. Do not reorder that.

**File descriptors.** A DataLoader over the 71,237-utterance eval set exhausts
the compute nodes' 1024 FD limit. The code sets the `file_system` sharing
strategy and the sbatch raises `ulimit -n`. Both are needed.

**Seed-suffixed output.** Multi-seed sweeps write to `runs*/<name>_s<seed>/`
(seed 1000 is unsuffixed). Any tool reading the unsuffixed path reports seed
1000's numbers for every seed — which once produced "results" identical to the
digit across different seeds.

---

## Statistics — non-negotiable

**Pair the arms.** Both configurations in a comparison must use the **same
seed**, so initialisation and data order are identical and shared variance
cancels. Measured on this project: raw EER swings ~1.2 pp between seeds, while
the paired difference is far tighter. Unpaired needs roughly 4x the runs.

**Never quote a power estimate from n<5.** At four pairs this project reported
"~9 seeds needed"; the fifth pair tripled the standard deviation and the true
figure was **266**. Every quantity derived from a small-sample sd inherits that
fragility.

**Wilcoxon cannot reach p<0.05 at n=5** — its minimum two-sided p is 0.0625. A
distribution-free result needs >=6 seeds.

**Seeds fix variance, not bias.** Every mistake this project made was
systematic. A hundred seeds would have reproduced each one faithfully and
returned a tight interval around a wrong number.

Analysis entry points: `scripts/paired_analysis.py`,
`scripts/score_analysis.py` (min t-DCF, per-attack), `scripts/tdcf_summary.py`.

---

## HPC conventions (shared `sameera` account)

Data and env:
- ASVspoof2019 LA: `/home/sameera/unofficail_audio_deepfake/Dataset/LA/`
  (train 25,380 / dev 24,844 / eval 71,237, all protocols present). Other
  candidate paths on the cluster are **incomplete** — do not use them.
- Python: `/home/sameera/.conda/envs/myenv/bin/python` (3.10, torch 2.9.1+cu128,
  soundfile). No pytest there, and **do not install into it** — it is shared.
- Project root on HPC: `/home/sameera/WorkBench/KA/safe/fsat`.

Scheduling:
- **`qos_gpu_h100` caps this account at `cpu=8` partition-wide.** A
  `--cpus-per-task=8` request serialises a whole array to one task at a time.
  Use 2.
- Per-user QOS: h100 3 jobs / 3 GPUs; h200 2 jobs; a100 2 jobs / 2 GPUs;
  rtx_pro_6000 2 GPUs.
- **Multi-partition requests are rejected** by this cluster's association. One
  partition per job; spreading means separate submissions. `scontrol update
  partition=a,b` fails too, and changing a job's partition requires changing its
  QOS to match or it sits on `Reason=InvalidQOS`.
- Measured runtimes for one training task: **H100 ~2h40m, Blackwell ~3h20m,
  V100 ~13h20m**. V100 has no TF32 and is ~5x slower here — a faster start is
  not a faster finish.

Multi-person etiquette:
- **Never cancel or mutate a job you did not submit** without the owner's
  explicit say-so. That includes the other collaborator's jobs and other
  students' work on this shared account.
- The `sal90rep_*`, `cvoice_*`, `as21_*`, `fp32_*` job families are SafeEar
  paper work and take precedence over this project. Let F-SAT jobs queue rather
  than compete.
- Prefix job names per person (`fsat_<initials>_*`) and use separate run
  directories, so nothing is ambiguous in `squeue` or on disk.
- Check what already shares the QOS cap before launching a sweep. Two efforts
  on one 3-slot budget serialise each other silently.
- There is a sibling ablation tree at `../FSAT-Exps` (band / gamma / pretrained
  sweeps) that imports this repo's `src/` read-only and shares the same QOS cap.

---

## Working in this repo

- `uv sync` then `uv run pytest -q` — 165 tests. Keep them passing.
- `scripts/smoke_test.py` runs the whole method on synthetic audio, no dataset
  and no GPU. Use it to check a change before spending cluster time.
- **No results in the README.** Numbers go in `RESULTS.md`, with the seed count
  and confidence interval stated. A number without its uncertainty is not a
  result here.
- Keep the "mistakes" section of `RESULTS.md` current. It is the most useful
  part of this repository for anyone learning to reproduce papers, and being
  wrong in public is the point.
- Use branches and PRs. Two people and two Claude sessions on one `main` has
  already caused concurrent-edit collisions in this tree.
