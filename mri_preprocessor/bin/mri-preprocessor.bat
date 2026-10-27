@echo off
REM mri-preprocessor launcher for Windows (PR19)
REM
REM Works in three scenarios:
REM 1. Bundled Python (standalone distribution)
REM 2. Local venv (.venv\)
REM 3. System Python (must be on PATH)

setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."

REM 1. Check for bundled Python
if exist "%APP_DIR%\runtime\python.exe" (
    "%APP_DIR%\runtime\python.exe" -m mri_preprocessor %*
    goto :eof
)

REM 2. Check for local venv
if exist "%APP_DIR%\.venv\Scripts\python.exe" (
    "%APP_DIR%\.venv\Scripts\python.exe" -m mri_preprocessor %*
    goto :eof
)

REM 3. Fall back to system Python
where python >nul 2>&1
if %errorlevel%==0 (
    python -m mri_preprocessor %*
    goto :eof
)

echo ERROR: No Python interpreter found.
echo   - For standalone: extract the full distribution (includes runtime\)
echo   - For development: run setup_windows.bat to create a venv
echo   - Or install Python 3.10+ and run: pip install .
exit /b 1
