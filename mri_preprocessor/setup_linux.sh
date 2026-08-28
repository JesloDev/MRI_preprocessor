#!/usr/bin/env bash
# Sets up the MRI Preprocessing Platform on a Linux machine using a
# standard Python venv (no conda required for development/testing).
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "== MRI Preprocessing Platform: Linux setup =="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found. Install Python 3.10+ first (e.g. sudo apt install python3 python3-venv)."
    exit 1
fi

echo "-- Creating virtual environment in $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "-- Upgrading pip"
pip install --upgrade pip

echo "-- Installing requirements (this can take a few minutes, SimpleITK/dipy are large)"
pip install -r requirements.txt

echo "-- Running test suite"
pytest -q

echo ""
echo "Setup complete. To start the server:"
echo "  source $VENV_DIR/bin/activate"
echo "  ./run_server.sh"
