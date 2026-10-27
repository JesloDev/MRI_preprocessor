"""
Tests for PR19: standalone application packaging.

Covers:
  - `python -m mri_preprocessor` entry point works
  - `bin/mri-preprocessor` launcher script works
  - pyproject.toml has correct metadata
  - __init__.py has __version__
  - The CLI works from the module entry point (not just `python -m app.cli`)
"""
import os
import subprocess
import sys

import pytest


# --------------------------------------------------------------------- #
# Package metadata
# --------------------------------------------------------------------- #

def test_package_has_version():
    """The package __init__.py exposes __version__."""
    import mri_preprocessor
    assert hasattr(mri_preprocessor, "__version__")
    assert mri_preprocessor.__version__ == "1.0.0"


def test_pyproject_toml_exists():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject = os.path.join(project_root, "pyproject.toml")
    assert os.path.isfile(pyproject)


def test_pyproject_has_entry_points():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject = os.path.join(project_root, "pyproject.toml")
    with open(pyproject) as f:
        content = f.read()
    assert "[project.scripts]" in content
    assert "mri-preprocessor" in content


def test_pyproject_has_dependencies():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject = os.path.join(project_root, "pyproject.toml")
    with open(pyproject) as f:
        content = f.read()
    assert "flask" in content
    assert "nibabel" in content
    assert "SimpleITK" in content
    assert "typer" in content
    assert "pyyaml" in content


# --------------------------------------------------------------------- #
# Module entry point: python -m mri_preprocessor
# --------------------------------------------------------------------- #

def _find_project_root():
    """Walk up from this test file to find the project root (the
    directory containing __main__.py and pyproject.toml)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def test_python_m_module_help():
    """`python -m mri_preprocessor --help` should produce the CLI help."""
    project_root = _find_project_root()
    parent = os.path.dirname(project_root)
    result = subprocess.run(
        [sys.executable, "-m", "mri_preprocessor", "--help"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": parent},
    )
    assert result.returncode == 0
    assert "MRI" in result.stdout
    assert "scan" in result.stdout
    assert "run" in result.stdout
    assert "engines" in result.stdout
    assert "presets" in result.stdout


def test_python_m_module_scan():
    """`python -m mri_preprocessor scan <dir>` should work end-to-end."""
    project_root = _find_project_root()
    parent = os.path.dirname(project_root)
    result = subprocess.run(
        [sys.executable, "-m", "mri_preprocessor", "scan", "/tmp/pr19_data"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": parent},
    )
    assert result.returncode == 0
    assert "Subjects: 1" in result.stdout
    assert "T1" in result.stdout


def test_python_m_module_engines_list():
    """`python -m mri_preprocessor engines list` should list all 6 engines."""
    project_root = _find_project_root()
    parent = os.path.dirname(project_root)
    result = subprocess.run(
        [sys.executable, "-m", "mri_preprocessor", "engines", "list"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": parent},
    )
    assert result.returncode == 0
    assert "synthstrip" in result.stdout
    assert "mcflirt" in result.stdout
    assert "fsl_eddy" in result.stdout


def test_python_m_module_capabilities():
    """`python -m mri_preprocessor capabilities` should list all steps."""
    project_root = _find_project_root()
    parent = os.path.dirname(project_root)
    result = subprocess.run(
        [sys.executable, "-m", "mri_preprocessor", "capabilities"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": parent},
    )
    assert result.returncode == 0
    assert "reorientation" in result.stdout
    assert "brain_extraction" in result.stdout
    assert "defacing" in result.stdout
    assert "anonymization" in result.stdout


# --------------------------------------------------------------------- #
# Launcher script
# --------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform == "win32",
                     reason="Bash launcher not available on Windows")
def test_launcher_script_help():
    """The bin/mri-preprocessor launcher should produce CLI help."""
    project_root = _find_project_root()
    launcher = os.path.join(project_root, "bin", "mri-preprocessor")
    assert os.path.isfile(launcher)
    assert os.access(launcher, os.X_OK)

    result = subprocess.run(
        [launcher, "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "MRI" in result.stdout
    assert "scan" in result.stdout


@pytest.mark.skipif(sys.platform == "win32",
                     reason="Bash launcher not available on Windows")
def test_launcher_script_engines_list():
    """The launcher should be able to run engines list."""
    project_root = _find_project_root()
    launcher = os.path.join(project_root, "bin", "mri-preprocessor")
    result = subprocess.run(
        [launcher, "engines", "list"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "synthstrip" in result.stdout


# --------------------------------------------------------------------- #
# File structure
# --------------------------------------------------------------------- #

def test_main_entry_point_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "__main__.py"))


def test_init_py_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "__init__.py"))


def test_launcher_script_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "bin", "mri-preprocessor"))


def test_launcher_bat_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "bin", "mri-preprocessor.bat"))


def test_pyinstaller_spec_exists():
    project_root = _find_project_root()
    spec = os.path.join(project_root, "packaging", "mri_preprocessor.spec")
    assert os.path.isfile(spec)


def test_packaging_docs_exists():
    project_root = _find_project_root()
    docs = os.path.join(project_root, "docs", "PACKAGING.md")
    assert os.path.isfile(docs)


def test_setup_windows_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "setup_windows.bat"))


def test_setup_linux_exists():
    project_root = _find_project_root()
    assert os.path.isfile(os.path.join(project_root, "setup_linux.sh"))


# --------------------------------------------------------------------- #
# CLI still works from the old entry point
# --------------------------------------------------------------------- #

def test_old_entry_point_still_works():
    """`python -m app.cli` should still work (backward compat)."""
    project_root = _find_project_root()
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=project_root,
    )
    assert result.returncode == 0
    assert "scan" in result.stdout
