#!/bin/bash
#SBATCH --job-name=fsat_paper
#SBATCH --array=2,3
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#
# Table 3 rows reproduced with the AUTHORS' OWN configuration, recovered from
# the ICLR 2025 supplementary archive (proceedings.iclr.cc, the paper links it
# nowhere and there is no GitHub release).
#
# Every value below is taken from their released code, not inferred from the
# paper text. Our first attempt guessed these from the prose and got ten of
# them wrong, which is why its result had to be withdrawn:
#
#   hyperparameters.py   lr 1e-5, epochs 10, bs 32, gamma 0.1,
#                        warmup_epochs 1, warmup_lr 1e-6, min_lr 1e-7,
#                        spectrum_epsilon 0.005, spectrum_alpha 0.002,
#                        spectrum_attack_iters 2, spectrum_restarts 1,
#                        spectrum_target_freq 4000,
#                        epsilon 5e-4, pgd-alpha 1e-4, attack_iters 2,
#                        aug_num 1, aug_prob 1.0, max_frames 600
#   frequency_attack.py  n_fft 1024, hop 512, Hann, no magnitude clamp
#   eval_pgd.py          eval eps 5e-4, alpha 1e-4, restarts 5, iters 10
#   train_*.py:35        fine-tunes a pretrained RawNet3 `model.pt`
#
# The pretrained checkpoint is the public RawNet3 speaker-verification model
# (huggingface jungjee/RawNet3, MIT). 226/234 tensors map onto this
# reimplementation once the `backbone.` prefix is applied; --min-pretrained-frac
# aborts if that silently stops being true.
#
#   FSAT_PRETRAINED=/path/to/model.pt sbatch sbatch/fsat_as19_paper.sbatch

set -uo pipefail

ROOT=${FSAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${FSAT_PYTHON:-python3}
MAN=$ROOT/manifests/asvspoof19
PRETRAINED=${FSAT_PRETRAINED:-$ROOT/model_zoo/rawnet3_model.pt}
SEED=${SEED:-1000}

# Tuning-sweep overrides. Defaults reproduce the authors' configuration exactly,
# so leaving these unset gives the paper-faithful run.
#   GAMMA  weight on the robust loss (authors: 0.1)
#   F_LO/F_HI  adversarial band in Hz (authors: 4000-8000)
#   TAG    when set, results go to runs_tuning/<TAG>_s<SEED> instead of
#          runs_paper/, so tuning variants never collide with the main sweep.
GAMMA=${GAMMA:-0.1}
F_LO=${F_LO:-4000}
F_HI=${F_HI:-8000}
TAG=${TAG:-}

NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)
ADVERSARY=(none none time fsat)
RANDAUG=(0 1 1 1)

# Per-domain TRAINING budgets, from hyperparameters.py. Frequency and time are
# separate parameters in their code, not two readings of one number.
TRAIN_EPS=(0.005 0.005 0.0005 0.005)
TRAIN_ALPHA=(0.002 0.002 0.0001 0.002)

i=${SLURM_ARRAY_TASK_ID:-3}
NAME=${NAMES[$i]}
ADV=${ADVERSARY[$i]}
AUG=${RANDAUG[$i]}

if [ ! -f "$PRETRAINED" ]; then
  echo "ERROR: pretrained checkpoint not found: $PRETRAINED" >&2
  exit 1
fi

if [ -n "$TAG" ]; then
  OUT=$ROOT/runs_tuning/${TAG}_s${SEED}
elif [ "$SEED" = "1000" ]; then
  OUT=$ROOT/runs_paper/$NAME
else
  OUT=$ROOT/runs_paper/${NAME}_s${SEED}
fi
mkdir -p "$OUT" "$ROOT/logs"

echo "=================================================================="
echo "config     : $NAME  (adversary=$ADV randaug=$AUG)  seed=$SEED"
echo "tuning     : gamma=$GAMMA band=${F_LO}-${F_HI}Hz tag=${TAG:-none}"
echo "pretrained : $PRETRAINED"
echo "budget     : train eps=${TRAIN_EPS[$i]} alpha=${TRAIN_ALPHA[$i]} iters=2"
echo "             eval  eps=5e-4 alpha=1e-4 restarts=5 iters=10"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "=================================================================="

cd "$ROOT"
export PYTHONPATH=$ROOT/src
export PYTHONUNBUFFERED=1
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

ARGS=(
  --train-manifest "$MAN/train.tsv"
  --val-manifest   "$MAN/dev.tsv"
  --test-manifest  "$MAN/eval.tsv"
  --pretrained "$PRETRAINED" --min-pretrained-frac 0.9
  --sample-rate 16000
  # max_frames 600 in their dataset -> 600*160+240 samples at 16 kHz ~= 6.0 s.
  --duration 6.0
  --batch-size 32
  --workers "${SLURM_CPUS_PER_TASK:-2}"
  --balanced-sampler
  --adversary "$ADV"
  --gamma "$GAMMA"
  --f-lo "$F_LO" --f-hi "$F_HI"
  --n-fft 1024 --hop-length 512
  --epsilon "${TRAIN_EPS[$i]}" --alpha "${TRAIN_ALPHA[$i]}" --steps 2 --restarts 1
  --eval-epsilon 5e-4 --eval-alpha 1e-4 --eval-alpha-time 1e-4
  --eval-steps 10 --eval-restarts 5
  --epochs 10
  --lr 1e-5
  --warmup-epochs 1 --warmup-lr 1e-6 --min-lr 1e-7
  --seed "$SEED"
  --device cuda
  --save   "$OUT/model.pt"
  --save-best
  --report "$OUT/report.json"
  --dump-scores "$OUT/scores.tsv"
  --eval-attacks --eval-corruptions --eval-subset 4000
  --log-every 50
)
# aug_num 1, aug_prob 1.0 in their code, not our earlier 2 / 0.5.
[ "$AUG" = "1" ] && ARGS+=(--randaug --randaug-n 1 --randaug-p 1.0)

srun "$PY" -u -m fsat.cli "${ARGS[@]}"
rc=$?

echo "=================================================================="
echo "finished   : $(date -Is)  rc=$rc"
echo "=================================================================="
exit $rc
