#!/usr/bin/env bash
# Starts the MRI Preprocessor REST API server (PR19).
#
# Works in the same three scenarios as bin/mri-preprocessor:
# 1. Bundled Python (standalone distribution)
# 2. Local venv (.venv/)
# 3. System Python
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MRIPREP_HOST="${MRIPREP_HOST:-127.0.0.1}"
export MRIPREP_PORT="${MRIPREP_PORT:-5000}"

# 1. Bundled Python
if [ -x "$SCRIPT_DIR/runtime/bin/python3" ]; then
    exec "$SCRIPT_DIR/runtime/bin/python3" -m app.main
fi

# 2. Local venv
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python3" -m app.main
fi

# 3. System Python (or active venv)
exec python3 -m app.main
