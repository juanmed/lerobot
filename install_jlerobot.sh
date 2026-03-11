#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

python -m pip install -e .
python -m pip install dynamixel-sdk safetensors
python -m pip install torch torchvision torchcodec --index-url https://download.pytorch.org/whl/cpu

if ! command -v spd-say >/dev/null 2>&1; then
  echo "Note: install 'speech-dispatcher' at the system level if you want voice prompts (spd-say)."
fi

echo "jlerobot install complete."
