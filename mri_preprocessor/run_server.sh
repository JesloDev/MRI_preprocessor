#!/usr/bin/env bash
set -euo pipefail
export MRIPREP_HOST="${MRIPREP_HOST:-127.0.0.1}"
export MRIPREP_PORT="${MRIPREP_PORT:-5000}"
python -m app.main
