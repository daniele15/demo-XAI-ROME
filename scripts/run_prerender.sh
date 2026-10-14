#!/usr/bin/env bash
# Background prerender: Full-mode tracing for 3 prompts x 3 kinds, then the
# Eiffel edit, then fallback.html. Resumable -- rerun and it skips what exists.
#
#   nohup ./scripts/run_prerender.sh &
#
# Progress:  tail -f prerendered/prerender.log
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p prerendered
LOG=prerendered/prerender.log
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== prerender run started (pid $$) ====="
  ./venv/bin/python scripts/prerender_trace.py
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] trace stage exit=$?"

  if [ ! -f prerendered/edit_eiffel.json ]; then
    ./venv/bin/python scripts/prerender_edit.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] edit stage exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP edit (prerendered/edit_eiffel.json exists)"
  fi

  if [ -f prerendered/trace_summary.json ] && [ -f prerendered/edit_eiffel.json ]; then
    ./venv/bin/python scripts/make_fallback.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] fallback stage exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP fallback (inputs incomplete)"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== prerender run finished ====="
} >> "$LOG" 2>&1
