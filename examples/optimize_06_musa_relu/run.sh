#!/usr/bin/env bash
# examples/optimize_06_musa_relu/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export OPENAI_MODEL=deepseek-chat
# export OPENAI_BASE_URL=https://api.deepseek.com/v1
# : "${OPENAI_API_KEY:?set OPENAI_API_KEY before running this example}"

export TORCH_MUSA_ARCH_LIST="${TORCH_MUSA_ARCH_LIST:-31}"
exec bash "$SCRIPT_DIR/../run_ppt_demo.sh" \
  --problem "$SCRIPT_DIR/problem.py" \
  --allow-third-party-api \
  --reasoning-effort medium \
  --generation-rounds 2 \
  --output-root "$SCRIPT_DIR/results" \
  "$@"
