#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multi.sh — Multi-seed / walk-forward launcher
#
# Usage:
#   ./run_multi.sh                    # seeds 42 123 456 on fold 0 (variance check)
#   ./run_multi.sh --folds 1 2 3      # seeds 42 123 456 on rolling folds 1-3
#   ./run_multi.sh --seeds 42 123     # custom seeds on fold 0
#   ./run_multi.sh --folds 1 2 3 --seeds 42 123 456  # full walk-forward
#
# Results are aggregated into logs_stable/wf_summary.csv when all runs finish.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON="/home/nosferatu/anaconda3/envs/freqtrade/bin/python"
CUDA_DEV=${CUDA_VISIBLE_DEVICES:-1}
SUMMARY="logs_stable/wf_summary.csv"

SEEDS=(42 123 456)
FOLDS=(0)

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)  shift; SEEDS=();  while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        --folds)  shift; FOLDS=();  while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  Multi-seed walk-forward launcher"
echo "  Seeds : ${SEEDS[*]}"
echo "  Folds : ${FOLDS[*]}"
echo "  GPU   : $CUDA_DEV"
echo "═══════════════════════════════════════════════════════════"

TOTAL=$(( ${#SEEDS[@]} * ${#FOLDS[@]} ))
DONE=0

for FOLD in "${FOLDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        DONE=$(( DONE + 1 ))
        LOG="logs_stable/train_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
        echo ""
        echo "── Run $DONE/$TOTAL  seed=$SEED  fold=$FOLD ──────────────────────────"
        echo "   Log: $LOG"

        CUDA_VISIBLE_DEVICES=$CUDA_DEV \
            $PYTHON train2.py --seed "$SEED" --fold "$FOLD" 2>&1 | tee "$LOG"

        echo "   ✓ Finished seed=$SEED fold=$FOLD"
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All $TOTAL runs complete. Aggregating results..."
echo "═══════════════════════════════════════════════════════════"

# Aggregate metrics.json from all runs into a CSV summary
$PYTHON - <<'PYEOF'
import json, glob, csv, sys
from pathlib import Path

rows = []
for mf in sorted(glob.glob("logs_stable/Run_*/metrics.json")):
    run = Path(mf).parent.name
    try:
        m = json.load(open(mf))
        m["run"] = run
        rows.append(m)
    except Exception as e:
        print(f"  Skip {mf}: {e}")

if not rows:
    print("  No metrics.json files found.")
    sys.exit(0)

keys = ["run", "seed", "fold", "best_val_sharpe", "best_val_return",
        "best_val_dd", "best_val_trades", "gen_ratio",
        "test_sharpe", "test_return", "test_dd"]
keys = [k for k in keys if any(k in r for r in rows)]

out = "logs_stable/wf_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"\n  Summary written to: {out}")
print(f"  {'Run':<20} {'ValSharpe':>10} {'ValRet':>8} {'GenR':>6} {'TestSharpe':>10}")
print(f"  {'─'*58}")
for r in rows:
    print(f"  {r.get('run','?'):<20} "
          f"{r.get('best_val_sharpe', float('nan')):>10.3f} "
          f"{r.get('best_val_return', float('nan'))*100:>7.1f}% "
          f"{r.get('gen_ratio', float('nan')):>6.2f} "
          f"{r.get('test_sharpe', float('nan')):>10.3f}")

import numpy as np
sharpes = [r.get("best_val_sharpe", float("nan")) for r in rows]
valid   = [s for s in sharpes if not (s != s)]  # filter NaN
if valid:
    print(f"\n  Val Sharpe  mean={np.mean(valid):.3f}  std={np.std(valid):.3f}  "
          f"{'✓ stable' if np.std(valid) < 0.5 else '✗ high variance'}")
PYEOF
