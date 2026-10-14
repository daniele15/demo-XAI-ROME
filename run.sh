#!/usr/bin/env bash
# Launch the ROME Gradio demo.
#   ./run.sh              -> http://localhost:7860
#   SHARE=1 ./run.sh      -> also create a public gradio.live link
set -euo pipefail
cd "$(dirname "$0")"

# rome/ is gitignored (it is upstream's repo, not ours), so a fresh clone of
# this project will not have it. Fetch it rather than failing on import.
if [ ! -d rome/.git ] && [ ! -f rome/globals.yml ]; then
  echo "rome/ missing -> cloning kmeng01/rome ..."
  git clone --depth 1 https://github.com/kmeng01/rome ./rome
fi

if [ ! -x venv/bin/python ]; then
  echo "venv missing. Create it with:" >&2
  echo "  python3 -m venv venv && ./venv/bin/python -m pip install \\" >&2
  echo "    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt" >&2
  exit 1
fi

# This box has 16 physical cores; leaving threads unset lets torch oversubscribe
# the 22 logical CPUs and actually run slower.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export SHARE="${SHARE:-0}"
export SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-7860}"

echo "ROME demo | model=${ROME_MODEL:-gpt2-xl} | SHARE=$SHARE | port=$SERVER_PORT"
exec ./venv/bin/python app.py
