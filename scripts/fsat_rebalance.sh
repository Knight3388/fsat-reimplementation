#!/bin/bash
# Hourly rebalancer for THIS PROJECT'S pending F-SAT jobs.
#
# Standing rule from the user (2026-08-12): a job pending more than 4 hours may
# be moved off its requested partition, even if that partition was explicitly
# asked for.
#
# Scope is deliberately narrow. It only ever touches a job that passes ALL of:
#   1. job name matches ^fsat_        (this project's prefix)
#   2. WorkDir is under the fsat tree (ground truth, names can collide)
#   3. state is PENDING               (never a running job)
#   4. pending longer than MIN_PEND_H (default 4h)
# Anything failing any gate is left alone. That deliberately excludes the
# SafeEar families (cvcomb_*, cvoice_*, as19_codec_*, as21_*, sal90rep_*,
# fp32_*), the sibling FSAT-Exps tree (fsatexp_*), and every other user on this
# shared account — notably abhijitdas's FSAT-A* jobs.
#
# It is also SPECULATION-FREE: a job is moved only to a partition with a
# genuinely free GPU *right now* and spare room under our per-user QOS cap, so
# the move converts directly into a start. Moving a job to another queue to
# wait is churn, and churn on a shared cluster is how you annoy people.
#
#   DRY_RUN=1 bash scripts/fsat_rebalance.sh      # default: log only
#   DRY_RUN=0 bash scripts/fsat_rebalance.sh      # actually move
#
# Install (hpc01):
#   0 * * * * DRY_RUN=0 /bin/bash /home/sameera/WorkBench/KA/safe/fsat/scripts/fsat_rebalance.sh \
#     >> /home/sameera/WorkBench/KA/safe/fsat/logs/rebalance.log 2>&1

set -uo pipefail

DRY_RUN="${DRY_RUN:-1}"
FSAT_ROOT="${FSAT_ROOT:-/home/sameera/WorkBench/KA/safe/fsat}"
NAME_RE='^fsat_'                       # gate 1
WORKDIR_PREFIX="$FSAT_ROOT"            # gate 2
MIN_PEND_H="${MIN_PEND_H:-4}"          # gate 4, the user's threshold
MAX_MOVES="${MAX_MOVES:-3}"            # churn cap per run
COOLDOWN_H="${COOLDOWN_H:-6}"          # do not move the same job twice quickly
STATE="${STATE:-$FSAT_ROOT/logs/.rebalance_state}"

# Destination preference, fastest first. V100 is last deliberately: measured
# ~5x slower here (13h16m vs 2h38m for the same job), so a faster queue start
# there is not a faster finish.
PARTS=(gpu_h100_4 gpu_rtx_pro_6000_6_csis_hyd gpu_a100_8 gpu_h200_8 gpu_v100_2)

# Per-user GPU cap per partition, from sacctmgr. Kept explicit so a silent
# cluster-side change shows up as a wrong decision in the log rather than a
# mysterious InvalidQOS.
declare -A PART_GPUCAP=(
  [gpu_h100_4]=3 [gpu_rtx_pro_6000_6_csis_hyd]=2
  [gpu_a100_8]=2 [gpu_h200_8]=3 [gpu_v100_2]=4
)
# This cluster binds a QOS per partition. Change partition without this and the
# job sits on Reason=InvalidQOS forever.
declare -A PART_QOS=(
  [gpu_h100_4]=qos_gpu_h100 [gpu_rtx_pro_6000_6_csis_hyd]=qos_gpu_rtx_pro_6000
  [gpu_a100_8]=qos_gpu_a100 [gpu_h200_8]=qos_gpu_h200 [gpu_v100_2]=gpulimit
)
# Walltime by job family and destination speed. Oversized limits hurt backfill.
walltime_for() {  # $1=jobname $2=partition
  local slow=0
  [[ "$2" == "gpu_v100_2" ]] && slow=1
  if [[ "$1" == fsat_rm* ]]; then          # 15 epochs @ bs16, ~2x the standard
    [[ $slow == 1 ]] && echo "2-00:00:00" || echo "12:00:00"
  else
    [[ $slow == 1 ]] && echo "1-00:00:00" || echo "06:00:00"
  fi
}

mkdir -p "$(dirname "$STATE")"; touch "$STATE"
log() { echo "[$(date -Is)] $*"; }

log "=== fsat_rebalance start (DRY_RUN=$DRY_RUN, threshold ${MIN_PEND_H}h) ==="

# --- free GPUs per partition, from AllocTRES (not node STATE, which reads
# --- "mix" whether 1 or all GPUs are busy) ---
declare -A FREE
for p in "${PARTS[@]}"; do
  free=0
  for n in $(sinfo -h -p "$p" -o "%N" | tr ',' ' ' | xargs -n1 scontrol show hostnames 2>/dev/null); do
    tot=$(scontrol show node "$n" 2>/dev/null | grep -oE 'Gres=gpu:[a-z0-9_.-]+:[0-9]+' | grep -oE '[0-9]+$' | head -1)
    alloc=$(scontrol show node "$n" 2>/dev/null | grep -oE 'AllocTRES=[^ ]*' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+$' | head -1)
    free=$(( free + ${tot:-0} - ${alloc:-0} ))
  done
  # subtract what we already hold there, against our per-user cap
  ours=$(squeue -u "$USER" -h -t R -p "$p" -o "%i" 2>/dev/null | wc -l)
  headroom=$(( ${PART_GPUCAP[$p]:-0} - ours ))
  (( headroom < 0 )) && headroom=0
  (( free > headroom )) && free=$headroom
  FREE[$p]=$free
  log "  capacity $p: $free usable slot(s)"
done

moves=0
now=$(date +%s)

while IFS='|' read -r jid jname jsub; do
  [[ -z "${jid:-}" ]] && continue
  (( moves >= MAX_MOVES )) && { log "  move cap ${MAX_MOVES} reached, stopping"; break; }

  # gate 1: name
  [[ "$jname" =~ $NAME_RE ]] || continue

  # gate 2: WorkDir is ground truth — names can be reused across projects
  wd=$(scontrol show job "${jid%%_*}" 2>/dev/null | grep -oE 'WorkDir=[^ ]+' | head -1 | cut -d= -f2-)
  if [[ "$wd" != "$WORKDIR_PREFIX"* ]]; then
    log "  SKIP $jid ($jname): WorkDir '$wd' outside $WORKDIR_PREFIX"
    continue
  fi

  # gate 4: pending long enough
  sub=$(date -d "$jsub" +%s 2>/dev/null) || continue
  pend_h=$(( (now - sub) / 3600 ))
  (( pend_h < MIN_PEND_H )) && continue

  # cooldown
  last=$(grep -E "^$jid " "$STATE" 2>/dev/null | tail -1 | awk '{print $2}')
  if [[ -n "${last:-}" ]] && (( (now - last) < COOLDOWN_H * 3600 )); then
    log "  SKIP $jid: moved within ${COOLDOWN_H}h, cooling down"
    continue
  fi

  cur=$(squeue -h -j "$jid" -o "%P" 2>/dev/null | head -1)
  for p in "${PARTS[@]}"; do
    [[ "$p" == "$cur" ]] && continue
    (( ${FREE[$p]:-0} > 0 )) || continue

    wt=$(walltime_for "$jname" "$p")
    log "  MOVE $jid ($jname, pending ${pend_h}h): $cur -> $p  qos=${PART_QOS[$p]} time=$wt"
    if [[ "$DRY_RUN" == "0" ]]; then
      if scontrol update jobid="$jid" partition="$p" 2>&1 &&
         scontrol update jobid="$jid" qos="${PART_QOS[$p]}" 2>&1 &&
         scontrol update jobid="$jid" TimeLimit="$wt" 2>&1; then
        echo "$jid $now $p" >> "$STATE"
        FREE[$p]=$(( ${FREE[$p]} - 1 ))
        moves=$(( moves + 1 ))
      else
        log "    FAILED, leaving $jid where it was"
      fi
    else
      FREE[$p]=$(( ${FREE[$p]} - 1 )); moves=$(( moves + 1 ))
    fi
    break
  done
done < <(squeue -u "$USER" -h -t PD -o "%i|%j|%V" 2>/dev/null)

log "=== end, $moves move(s) ==="
