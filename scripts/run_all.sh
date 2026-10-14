#!/usr/bin/env bash
# Sequential: benchmark first (gives the UI its time estimates and the early
# Fast-mode result), then the Full-mode prerender. NOT parallel -- two copies
# of gpt2-xl fp32 would be ~12 GB and this box has ~11 GB free.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=prerendered/prerender.log
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== run_all started (pid $$) ====="
  if [ ! -f prerendered/benchmark.json ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- stage 1/4: benchmark ---"
    ./venv/bin/python scripts/benchmark.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] benchmark exit=$?"
    ./venv/bin/python scripts/update_readme_bench.py
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP benchmark (prerendered/benchmark.json exists)"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- stage 2/4: Full-mode tracing ---"
  ./venv/bin/python scripts/prerender_trace.py
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] trace exit=$?"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- stage 3/4: ROME edit ---"
  if [ ! -f prerendered/edit_eiffel.json ]; then
    ./venv/bin/python scripts/prerender_edit.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] edit exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP edit (already done)"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- stage 4/4: fallback.html ---"
  if [ -f prerendered/trace_summary.json ] && [ -f prerendered/edit_eiffel.json ]; then
    ./venv/bin/python scripts/make_fallback.py
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] fallback exit=$?"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP fallback (inputs incomplete)"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== run_all finished ====="
} >> "$LOG" 2>&1
