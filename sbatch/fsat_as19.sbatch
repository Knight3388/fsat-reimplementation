#!/bin/bash
#SBATCH --job-name=fsat_as19
#SBATCH --array=0-3
# Partition is site-specific. Override at submit time:
#   sbatch --partition=<name> sbatch/fsat_as19.sbatch
#
# Some clusters reject multi-partition requests ("Multiple partition job request
# not supported when a partition is set in the association"), in which case a
# job must name exactly ONE partition and spreading work means submitting
# separately per partition. Check your site's QOS caps (sacctmgr show qos) for
# per-user job, GPU and CPU limits before sizing an array: a cpus-per-task that
# exceeds the QOS CPU cap will serialise the whole array.
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# Relative to the submit directory, so no absolute site paths are baked in.
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
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

# Site configuration. Set these in your environment, or export them once in a
# site file and `sbatch --export=ALL`:
#   FSAT_ROOT    checkout directory (defaults to this script's parent)
#   FSAT_PYTHON  interpreter with torch + soundfile installed
ROOT=${FSAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${FSAT_PYTHON:-python3}
SMOKE=${SMOKE:-0}

# SEED controls both the run's randomness and its output directory, so a
# multi-seed sweep does not clobber itself. Use the SAME seed for both arms of a
# comparison: identical init and data order makes the difference paired, which
# cancels shared variance and needs roughly half the runs of an unpaired design.
SEED=${SEED:-1000}

# Table 3 rows, in order.
NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)
ADVERSARY=(none none time fsat)
RANDAUG=(0 1 1 1)

# Per-row training budget. The paper gives DIFFERENT budgets per domain:
# eps=0.01 / alpha=4e-4 in frequency, eps=1e-4 / alpha=4e-5 in time. Applying
# the frequency budget to a time-domain waveform attack over-perturbs it 100x
# and cripples the isotropic-AT baseline that F-SAT is compared against.
TRAIN_EPS=(0.01 0.01 1e-4 0.01)
TRAIN_ALPHA=(4e-4 4e-4 4e-5 4e-4)

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

# TEST_MAN can be overridden to point a smoke-sized training run at the FULL
# eval manifest, which is the only way to exercise the file-descriptor path
# that killed job 263770 without paying for a full training run.
TEST_MAN=${TEST_MAN:-$MAN/eval.tsv}

# Seed 1000 keeps the original unsuffixed path so the first batch of runs stays
# addressable; every other seed gets its own directory.
if [ "$SEED" = "1000" ]; then
  OUT=$ROOT/runs/$NAME
else
  OUT=$ROOT/runs/${NAME}_s${SEED}
fi
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

# Compute nodes default to a 1024 FD soft limit, which a DataLoader over the
# 71,237-utterance eval manifest exhausts. The code also sets the file_system
# sharing strategy; this is the second line of defence.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "open-file limit: $(ulimit -n)"

ARGS=(
  --train-manifest "$MAN/train.tsv"
  --val-manifest   "$MAN/dev.tsv"
  --test-manifest  "$TEST_MAN"
  --sample-rate 16000
  --duration 4.0
  --batch-size 32
  --workers "${SLURM_CPUS_PER_TASK:-2}"
  --balanced-sampler
  --adversary "$ADV"
  --gamma 0.1
  --f-lo 4000 --f-hi 8000
  --epsilon "${TRAIN_EPS[$i]}" --alpha "${TRAIN_ALPHA[$i]}" --steps 5
  --eval-epsilon 1e-4 --eval-alpha 4e-4 --eval-alpha-time 4e-5
  --epochs "$EPOCHS"
  --lr 1e-3
  --seed "$SEED"
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
