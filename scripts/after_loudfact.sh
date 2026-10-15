#!/usr/bin/env bash
# Sequential, one model in RAM at a time:
#   wait for pick_loud_fact -> measure the runner-up -> resume the 6 heatmaps.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false

while pgrep -f 'scripts/pick_loud_fact.py' > /dev/null; do sleep 15; done
echo "[$(date '+%H:%M:%S')] pick_loud_fact done -> runner-up" >> prerendered/loud_fact.log
./venv/bin/python scripts/loud_fact_second.py >> prerendered/loud_fact.log 2>&1
echo "[$(date '+%H:%M:%S')] runner-up done -> resuming heatmaps" >> prerendered/loud_fact.log
exec ./scripts/prerender_chain.sh
