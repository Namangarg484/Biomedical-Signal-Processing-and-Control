#!/bin/zsh
# Sequential driver v2 — finish scarcity, THEN preproc A (no concurrent GPU/mem
# allocation, which is what crashed the first scarcity run).
set -u
cd /Users/namangarg/Desktop/Raman
LOG=outputs/audit_experiments.log
echo "[driver2] started $(date)" >> "$LOG"

# 1) Complete scarcity (reuses 6 cached runs, trains remaining 6)
echo "[driver2] === Finishing scarcity === $(date)" >> "$LOG"
python scripts/scarcity_5class.py >> outputs/scarcity_5class/run.log 2>&1
echo "[driver2] scarcity rc=$? $(date)" >> "$LOG"

# 2) Preprocessing ablation A (bug fixed: wavelet='morl')
echo "[driver2] === Experiment A: preprocessing ablation === $(date)" >> "$LOG"
python scripts/preproc_ablation.py >> outputs/preproc_ablation_run.log 2>&1
echo "[driver2] A rc=$? $(date)" >> "$LOG"

echo "[driver2] DRIVER COMPLETE $(date)" >> "$LOG"
