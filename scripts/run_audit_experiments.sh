#!/bin/zsh
# Sequential experiment driver — runs after the scarcity job to avoid GPU contention.
# 1) wait for scarcity (PID passed as $1) to finish
# 2) Experiment A: preprocessing ablation (~30 min)
# 3) Experiment C: 30-species fusion comparison (~8-10 h)
set -u
cd /Users/namangarg/Desktop/Raman
LOG=outputs/audit_experiments.log
echo "[driver] started $(date)" > "$LOG"

SCARCITY_PID="${1:-}"
if [[ -n "$SCARCITY_PID" ]]; then
  echo "[driver] waiting for scarcity PID $SCARCITY_PID ..." >> "$LOG"
  while kill -0 "$SCARCITY_PID" 2>/dev/null; do sleep 30; done
  echo "[driver] scarcity finished $(date)" >> "$LOG"
fi

echo "[driver] === Experiment A: preprocessing ablation === $(date)" >> "$LOG"
python scripts/preproc_ablation.py >> "$LOG" 2>&1
echo "[driver] A done rc=$? $(date)" >> "$LOG"

echo "[driver] (Experiment C is run manually by user — skipped here) $(date)" >> "$LOG"
echo "[driver] DRIVER COMPLETE $(date)" >> "$LOG"
