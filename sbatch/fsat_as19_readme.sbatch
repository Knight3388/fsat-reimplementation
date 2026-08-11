#!/bin/bash
#SBATCH --job-name=fsat_readme
#SBATCH --array=3
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#
# Attempt 4: the authors' EXACT reproduce command, from readme.txt in the ICLR
# supplementary archive. This differs from hyperparameters.py's DEFAULTS, which
# is what sbatch/fsat_as19_paper.sbatch encodes:
#
#   python train_attack_frequency.py --lr 0.00001 --epochs 15 --mixup \
#     --mixup_alpha 0.5 --max_frames 600 --bs 16 --aug_num 1 --aug_prob 0.9 \
#     --attack spectrum --attack_iters 2 --spectrum_target_freq 4000 \
#     --spectrum_epsilon 0.01 --spectrum_alpha 0.005 --gamma 0.1
#
# Differences from the defaults we ran previously:
#   epochs   15  (not 10)
#   bs       16  (not 32)
#   eps    0.01  (not 0.005)
#   alpha 0.005  (not 0.002)
#   aug_prob 0.9 (not 1.0)
#   mixup    ON, alpha 0.5  (we had none at all)
#
# Mixup is not optional in their code: train_attack_frequency.py opens with
#   assert args.mixup, "Mixup is preferred for spectrum attack"
# so every earlier F-SAT run of ours was missing a component their
# implementation requires.
#
#   SEED=1000 sbatch sbatch/fsat_as19_readme.sbatch

set -uo pipefail

ROOT=${FSAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${FSAT_PYTHON:-python3}
MAN=$ROOT/manifests/asvspoof19
PRETRAINED=${FSAT_PRETRAINED:-$ROOT/model_zoo/rawnet3_model.pt}
SEED=${SEED:-1000}
TAG=${TAG:-readme}

if [ ! -f "$PRETRAINED" ]; then
  echo "ERROR: pretrained checkpoint not found: $PRETRAINED" >&2
  exit 1
fi

OUT=$ROOT/runs_tuning/${TAG}_s${SEED}
mkdir -p "$OUT" "$ROOT/logs"

echo "=================================================================="
echo "attempt 4  : authors' exact readme.txt command"
echo "seed       : $SEED"
echo "differences: epochs 15, bs 16, eps 0.01, alpha 0.005, mixup 0.5, aug_prob 0.9"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "=================================================================="

cd "$ROOT"
export PYTHONPATH=$ROOT/src
export PYTHONUNBUFFERED=1
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

srun "$PY" -u -m fsat.cli \
  --train-manifest "$MAN/train.tsv" \
  --val-manifest   "$MAN/dev.tsv" \
  --test-manifest  "$MAN/eval.tsv" \
  --pretrained "$PRETRAINED" --min-pretrained-frac 0.9 \
  --sample-rate 16000 --duration 6.0 \
  --batch-size 16 \
  --workers "${SLURM_CPUS_PER_TASK:-2}" \
  --balanced-sampler \
  --adversary fsat \
  --gamma 0.1 \
  --f-lo 4000 --f-hi 8000 \
  --n-fft 1024 --hop-length 512 \
  --epsilon 0.01 --alpha 0.005 --steps 2 --restarts 1 \
  --mixup-alpha 0.5 \
  --randaug --randaug-n 1 --randaug-p 0.9 \
  --eval-epsilon 5e-4 --eval-alpha 1e-4 --eval-alpha-time 1e-4 \
  --eval-steps 10 --eval-restarts 5 \
  --epochs 15 \
  --lr 1e-5 \
  --warmup-epochs 1 --warmup-lr 1e-6 --min-lr 1e-7 \
  --seed "$SEED" \
  --device cuda \
  --save   "$OUT/model.pt" --save-best \
  --report "$OUT/report.json" \
  --dump-scores "$OUT/scores.tsv" \
  --eval-attacks --eval-corruptions --eval-subset 4000 \
  --log-every 50
rc=$?

echo "=================================================================="
echo "finished   : $(date -Is)  rc=$rc"
echo "=================================================================="
exit $rc
