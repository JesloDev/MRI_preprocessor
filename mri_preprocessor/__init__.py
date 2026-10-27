"""
Package init for mri_preprocessor (PR19).

The package root is `mri_preprocessor/` (this directory). The `app/`
subdirectory contains the actual application code. This split allows:
  - `pip install .` to install `app` as a top-level package
  - `python -m mri_preprocessor` to run the CLI from a bundled
    distribution where `app` isn't pip-installed

The __version__ is used by the CLI `--version` flag and by PyInstaller
to stamp the build version.
"""
__version__ = "1.0.0"
