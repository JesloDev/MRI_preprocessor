@echo off
REM MRI Preprocessor: Windows setup script (PR19)
REM
REM Creates a Python virtual environment and installs all dependencies.

setlocal enabledelayedexpansion

set "PYTHON_BIN=python"
set "VENV_DIR=.venv"

echo ================================================
echo   MRI Preprocessor: Windows Setup
echo ================================================
echo

REM 1. Check Python
where %PYTHON_BIN% >nul 2>&1
if errorlevel 1 (
    echo ERROR: %PYTHON_BIN% not found.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   Make sure "Add Python to PATH" is checked during installation.
    exit /b 1
)

echo -- Creating virtual environment...
%PYTHON_BIN% -m venv %VENV_DIR%
call %VENV_DIR%\Scripts\activate.bat

echo -- Upgrading pip...
pip install --upgrade pip --quiet

echo -- Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet
pip install pyyaml --quiet

echo -- Installing MRI Preprocessor...
pip install -e . --quiet --no-deps

echo.
echo -- Running test suite...
pytest -q

echo.
echo ================================================
echo   Setup complete!
echo ================================================
echo.
echo   CLI:
echo     bin\mri-preprocessor.bat --help
echo     bin\mri-preprocessor.bat scan C:\path\to\data
echo     bin\mri-preprocessor.bat run C:\in C:\out --preset ml-ready
echo.
echo   REST API server:
echo     python -m app.main
echo.
echo   Engine management:
echo     bin\mri-preprocessor.bat engines list
echo     bin\mri-preprocessor.bat engines install synthstrip --bundle C:\path\to\bundle
echo.
