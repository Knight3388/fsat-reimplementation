#!/bin/bash
# Emit one line per array task that has reached a TERMINAL state, with its
# headline clean numbers when a report exists.
#
# Terminal covers failure states too, not just COMPLETED: a watcher that only
# matches success stays silent through a crash, and silence is indistinguishable
# from "still running".
#
# Usage: poll_status.sh <array_job_id>

JOB=${1:?usage: poll_status.sh <array_job_id>}
ROOT=/scratch/sameera/fsat
PY=/home/sameera/.conda/envs/myenv/bin/python
NAMES=(rawnet3 rawnet3_randaug rawnet3_randaug_at_time rawnet3_randaug_fsat)

sacct -j "$JOB" -X --noheader -P -o JobID,State,Elapsed 2>/dev/null |
while IFS='|' read -r id state elapsed; do
  case "$state" in
    COMPLETED|FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) ;;
    *) continue ;;
  esac

  idx=${id##*_}
  name=${NAMES[$idx]:-task$idx}
  report=$ROOT/runs/$name/report.json
  summary="no report written"

  if [ -f "$report" ]; then
    summary=$("$PY" - "$report" <<'PYEOF' 2>/dev/null || echo "report unreadable"
import json, sys
d = json.load(open(sys.argv[1]))
c = d["clean"]
eer = c.get("eer")
print("eer=%s real=%.1f%% fake=%.1f%% avg=%.1f%% n=%s best_epoch=%s" % (
    ("%.2f%%" % (eer * 100)) if eer is not None else "n/a",
    c["real_accuracy"] * 100, c["fake_accuracy"] * 100,
    c["average_accuracy"] * 100, d.get("report_n", "?"), d.get("best_epoch", "?")))
PYEOF
)
  fi

  printf '%s [%s] %s %s :: %s\n' "$id" "$state" "$elapsed" "$name" "$summary"
done | sort
