#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m playwright install chromium

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  if [[ -f "$ROOT_DIR/.env.example" ]]; then
    sed 's/your_api_key_here//g' "$ROOT_DIR/.env.example" > "$ROOT_DIR/.env"
  else
    printf 'LLM_PROVIDER=\n' > "$ROOT_DIR/.env"
  fi
fi

echo "Setup complete."
