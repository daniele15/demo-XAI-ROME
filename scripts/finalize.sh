#!/usr/bin/env bash
# Runs after the 9 heatmaps: regenerate the edit with the final 6-probe set and
# a fixed seed, rebuild fallback.html, refresh the README table.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=prerendered/prerender.log
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false

while pgrep -f 'prerender_chain.sh' > /dev/null; do sleep 20; done
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== finalize ====="
  # The existing JSON predates both the seed and the 6-probe set.
  rm -f prerendered/edit_eiffel.json
  ./venv/bin/python scripts/prerender_edit.py; echo "edit exit=$?"
  ./venv/bin/python scripts/make_fallback.py; echo "fallback exit=$?"
  ./venv/bin/python scripts/update_readme_bench.py; echo "readme exit=$?"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== finalize done ====="
} >> "$LOG" 2>&1
