#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export OPENAI_MODEL=deepseek-chat
# export OPENAI_BASE_URL=https://api.deepseek.com/v1
# export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY first}"

exec bash "$SCRIPT_DIR/../run_ppt_demo.sh" \
  --problem "$SCRIPT_DIR/problem.py" \
  --allow-third-party-api \
  --reasoning-effort medium \
  --generation-rounds 5 \
  --output-root "$SCRIPT_DIR/results" \
  "$@"
