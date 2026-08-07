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

# Evaluation budget. Defaults reproduce the eps=1e-4-everywhere reading.
# Set EPS_TIME different from EPS_FREQ for the DOMAIN-MATCHED reading, i.e.
# frequency probes at 0.01 and the time probe at 1e-4, which is the other
# defensible way to read the paper's two quoted epsilons.
EPS_FREQ=${EPS_FREQ:-1e-4}
EPS_TIME=${EPS_TIME:-$EPS_FREQ}
A_FREQ=${A_FREQ:-4e-4}
A_TIME=${A_TIME:-4e-5}
SEED=${SEED:-1000}
DST=${DST:-$ROOT/runs_eval_e${EPS_FREQ}_t${EPS_TIME}}

NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)

i=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$i]}
# Seed 1000 lives in the unsuffixed directory, every other seed is suffixed.
if [ "$SEED" = "1000" ]; then RUNDIR=$NAME; else RUNDIR=${NAME}_s${SEED}; fi
CKPT=$SRC/$RUNDIR/model.pt

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

OUT=$DST/$RUNDIR
mkdir -p "$OUT" "$ROOT/logs"

echo "=================================================================="
echo "re-evaluating : $RUNDIR (seed $SEED)"
echo "checkpoint    : $CKPT"
echo "budget        : eps_freq=$EPS_FREQ eps_time=$EPS_TIME alpha_freq=$A_FREQ alpha_time=$A_TIME"
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
  --eval-epsilon "$EPS_FREQ" --eval-epsilon-time "$EPS_TIME" \
  --eval-alpha "$A_FREQ" --eval-alpha-time "$A_TIME" \
  --steps 5 \
  --device cuda \
  --report "$OUT/report.json" \
  --eval-attacks --eval-corruptions --eval-subset 4000 \
  ${DUMP_SCORES:+--dump-scores "$OUT/scores.tsv"} \
  --seed "$SEED"
rc=$?

echo "=================================================================="
echo "finished      : $(date -Is)  rc=$rc"
echo "=================================================================="
exit $rc
