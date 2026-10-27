"""
Entry point for `python -m mri_preprocessor` (PR19).

This allows the application to be run as a module without needing
`pip install` -- useful for portable distributions where the Python
environment is bundled alongside the application code.

Usage:
    python -m mri_preprocessor scan /data
    python -m mri_preprocessor run /in /out --preset ml-ready
    python -m mri_preprocessor engines list

This file is a thin shim -- it just calls the Typer app from app.cli.
"""
import sys
import os

# Ensure the parent directory (containing the `app/` package) is on
# sys.path. When running `python -m mri_preprocessor`, Python adds
# the current directory to sys.path, so `app` is importable. But
# when running from a bundled environment where the CWD might be
# different, we need to make sure.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from app.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
