#!/usr/bin/env bash
# MRI Preprocessor: Linux/macOS setup script (PR19)
#
# Creates a Python virtual environment and installs all dependencies.
# After running this, the application can be started via:
#   ./bin/mri-preprocessor scan /data
#   ./bin/mri-preprocessor run /in /out --preset ml-ready
#   ./bin/mri-preprocessor-server  (REST API on http://127.0.0.1:5000)
#
# This script does NOT require conda or micromamba. It uses the
# standard Python venv module. The FSL/FreeSurfer external engines
# (if needed) are installed separately via:
#   ./bin/mri-preprocessor engines install synthstrip --bundle /path/to/bundle

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "================================================"
echo "  MRI Preprocessor: Setup"
echo "================================================"
echo ""

# 1. Check Python
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found."
    echo "  Install Python 3.10+ first:"
    echo "    Ubuntu/Debian: sudo apt install python3 python3-venv"
    echo "    macOS:        brew install python3"
    echo "    Or download:   https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $PYTHON_BIN ($PY_VERSION)"
echo "  Venv:   $VENV_DIR"
echo ""

# 2. Create venv
echo "-- Creating virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. Install deps
echo "-- Upgrading pip..."
pip install --upgrade pip --quiet

echo "-- Installing dependencies (SimpleITK/DIPY are large, this may take a few minutes)..."
pip install -r requirements.txt --quiet

echo "-- Installing pyyaml (for engine manifests)..."
pip install pyyaml --quiet

# 4. Install the application in development mode (provides console scripts)
echo "-- Installing MRI Preprocessor (development mode)..."
pip install -e . --quiet --no-deps

# 5. Run tests
echo ""
echo "-- Running test suite..."
pytest -q

# 6. Verify
echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "  CLI:"
echo "    ./bin/mri-preprocessor --help"
echo "    ./bin/mri-preprocessor scan /path/to/data"
echo "    ./bin/mri-preprocessor run /in /out --preset ml-ready"
echo ""
echo "  REST API server:"
echo "    ./run_server.sh    (starts on http://127.0.0.1:5000)"
echo ""
echo "  Engine management:"
echo "    ./bin/mri-preprocessor engines list"
echo "    ./bin/mri-preprocessor engines install synthstrip --bundle /path/to/bundle"
echo "    ./bin/mri-preprocessor engines health synthstrip"
echo ""
echo "  Preset installation (auto-installs missing engines):"
echo "    ./bin/mri-preprocessor presets install ml-ready --bundle-dir /path/to/bundles"
echo ""
