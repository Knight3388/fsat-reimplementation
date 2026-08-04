#!/bin/bash
#SBATCH --job-name=fsat_reeval
#SBATCH --array=0-3
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Relative to the submit directory, so no absolute site paths are baked in.
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#
# Re-evaluate existing checkpoints at the PAPER'S evaluation budget
# (Table 5: eps=1e-4, alpha=4e-4 frequency, alpha=4e-5 time).
#
# The first run swept attacks at eps=0.01, the training budget. That flatters
# F-SAT, whose training attack is exactly eps=0.01 magnitude perturbation in
# the 4-8 kHz band being probed. No retraining here, weights are loaded from
# the saved checkpoints.
#
#   sbatch sbatch/fsat_reeval.sbatch

set -uo pipefail

# See fsat_as19.sbatch for FSAT_ROOT / FSAT_PYTHON.
ROOT=${FSAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${FSAT_PYTHON:-python3}
MAN=$ROOT/manifests/asvspoof19
SRC=${SRC:-$ROOT/runs}
DST=${DST:-$ROOT/runs_eval1e-4}

NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)
RANDAUG=(0 1 1 1)

i=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$i]}
CKPT=$SRC/$NAME/model.pt

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

OUT=$DST/$NAME
mkdir -p "$OUT" "$ROOT/logs"

echo "=================================================================="
echo "re-evaluating : $NAME"
echo "checkpoint    : $CKPT"
echo "budget        : eps=1e-4 alpha_freq=4e-4 alpha_time=4e-5 (paper Table 5)"
echo "node          : $(hostname)"
echo "started       : $(date -Is)"
echo "=================================================================="

cd "$ROOT"
export PYTHONPATH=$ROOT/src
export PYTHONUNBUFFERED=1
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# --train-manifest is required for the manifest code path but no training runs
# under --eval-only, so it is only used to construct the dataset objects.
srun "$PY" -u -m fsat.cli \
  --train-manifest "$MAN/train.tsv" \
  --val-manifest   "$MAN/dev.tsv" \
  --test-manifest  "$MAN/eval.tsv" \
  --sample-rate 16000 --duration 4.0 --batch-size 32 \
  --workers "${SLURM_CPUS_PER_TASK:-2}" \
  --load "$CKPT" --eval-only \
  --eval-epsilon 1e-4 --eval-alpha 4e-4 --eval-alpha-time 4e-5 \
  --steps 5 \
  --device cuda \
  --report "$OUT/report.json" \
  --eval-attacks --eval-corruptions --eval-subset 4000 \
  --seed 1000
rc=$?

echo "=================================================================="
echo "finished      : $(date -Is)  rc=$rc"
echo "=================================================================="
exit $rc
