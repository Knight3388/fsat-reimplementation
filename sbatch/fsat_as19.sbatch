#!/bin/bash
#SBATCH --job-name=fsat_as19
#SBATCH --array=0-3
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
# qos_gpu_h100 caps this user at cpu=8 across the whole partition, so asking
# for 8 here would let only ONE array task run at a time. 2 keeps three
# concurrent, which is also the QOS ceiling on GPUs.
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/sameera/fsat/logs/%x_%A_%a.out
#SBATCH --error=/scratch/sameera/fsat/logs/%x_%A_%a.out
#
# Table 3 of arXiv:2411.00121v1, reproduced on ASVspoof2019 LA.
#
# The paper's own corpus (DeepFakeVox-HQ) was never released, so this targets
# ASVspoof2019, which the paper also evaluates on. Four array tasks, one per
# Table 3 row. These numbers are NOT comparable to the paper's Table 2
# headline, which is measured on DeepFakeVox-HQ.
#
# Model selection uses dev, all reported numbers come from eval. They are
# separate splits precisely so selection never sees the reporting set.
#
# Sanity pass first (tiny manifests, ~10 min):
#   sbatch --export=ALL,SMOKE=1 --time=00:40:00 sbatch/fsat_as19.sbatch
# Then the real run:
#   sbatch sbatch/fsat_as19.sbatch

set -uo pipefail

ROOT=/scratch/sameera/fsat
PY=/home/sameera/.conda/envs/myenv/bin/python
SMOKE=${SMOKE:-0}

# Table 3 rows, in order.
NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)
ADVERSARY=(none none time fsat)
RANDAUG=(0 1 1 1)

i=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$i]}
ADV=${ADVERSARY[$i]}
AUG=${RANDAUG[$i]}

if [ "$SMOKE" = "1" ]; then
  # Small randomly-sampled manifests so the smoke pass exercises the real code
  # path (flac decode, balanced sampler, best-checkpoint, sweeps) in minutes.
  # shuf keeps the class mix representative; a head slice would not.
  MAN=$ROOT/manifests/asvspoof19_smoke
  NAME="smoke_$NAME"
  EPOCHS=1
  EVAL_SUBSET=128
else
  MAN=$ROOT/manifests/asvspoof19
  EPOCHS=30
  EVAL_SUBSET=4000
fi

OUT=$ROOT/runs/$NAME
mkdir -p "$OUT" "$ROOT/logs"

echo "=================================================================="
echo "job        : ${SLURM_JOB_ID:-local} array task $i"
echo "config     : $NAME  (adversary=$ADV randaug=$AUG)"
echo "node       : $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "manifests  : $MAN"
echo "smoke=$SMOKE epochs=$EPOCHS sweep_subset=$EVAL_SUBSET"
echo "started    : $(date -Is)"
echo "=================================================================="

cd "$ROOT"
export PYTHONPATH=$ROOT/src
export PYTHONUNBUFFERED=1

ARGS=(
  --train-manifest "$MAN/train.tsv"
  --val-manifest   "$MAN/dev.tsv"
  --test-manifest  "$MAN/eval.tsv"
  --sample-rate 16000
  --duration 4.0
  --batch-size 32
  --workers "${SLURM_CPUS_PER_TASK:-2}"
  --balanced-sampler
  --adversary "$ADV"
  --gamma 0.1
  --f-lo 4000 --f-hi 8000
  --epsilon 0.01 --alpha 0.004 --steps 5
  --epochs "$EPOCHS"
  --lr 1e-3
  --seed 1000
  --device cuda
  --save      "$OUT/model.pt"
  --save-best
  --report    "$OUT/report.json"
  --eval-attacks
  --eval-corruptions
  --eval-subset "$EVAL_SUBSET"
  --log-every 50
)
[ "$AUG" = "1" ] && ARGS+=(--randaug --randaug-n 2 --randaug-p 0.5)

srun "$PY" -u -m fsat.cli "${ARGS[@]}"
rc=$?

echo "=================================================================="
echo "finished   : $(date -Is)  rc=$rc"
echo "report     : $OUT/report.json"
echo "=================================================================="
exit $rc
