#!/usr/bin/env bash
# Edit first (checkpoint 2 needs it), then the v-step decision, then the 9
# Full heatmaps, then fallback.html. Every stage is skippable/resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=prerendered/prerender.log
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== chain started pid $$ ====="

  if [ ! -f prerendered/edit_eiffel.json ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- A: ROME edit + 5 probes + restore check ---"
    ./venv/bin/python scripts/prerender_edit.py; echo "A exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP A"
  fi

  if [ ! -f prerendered/v_steps_comparison.json ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- B: Full vs reduced v_num_grad_steps ---"
    ./venv/bin/python scripts/compare_v_steps.py; echo "B exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP B"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- C: 9 Full heatmaps ---"
  ./venv/bin/python scripts/prerender_trace.py; echo "C exit=$?"

  if [ -f prerendered/trace_summary.json ] && [ -f prerendered/edit_eiffel.json ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- D: fallback.html ---"
    ./venv/bin/python scripts/make_fallback.py; echo "D exit=$?"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== chain finished ====="
} >> "$LOG" 2>&1
