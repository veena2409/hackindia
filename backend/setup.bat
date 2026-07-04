@echo off
REM ── MedGemma Backend Setup ────────────────────────────────────────────────
REM Run this once from the project root to install all Python dependencies.

echo.
echo  Setting up MedGemma backend...
echo.

REM 1. Copy .env.example to .env if it doesn't exist yet
if not exist backend\.env (
    copy backend\.env.example backend\.env
    echo  Created backend\.env from template.
    echo  IMPORTANT: Open backend\.env and fill in your real endpoint + token!
    echo.
) else (
    echo  backend\.env already exists - skipping copy.
)

REM 2. Install Python dependencies into backend\.deps
python -m pip install -r backend\requirements.txt --target backend\.deps --quiet
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: pip install failed. Make sure Python 3.10+ is on your PATH.
    exit /b 1
)

echo.
echo  All dependencies installed to backend\.deps
echo.
echo  To run the smoke test (text-only):
echo    set PYTHONPATH=backend\.deps
echo    python backend\medgemma_client.py
echo.
echo  To run with an MRI image:
echo    python backend\medgemma_client.py path\to\your_mri.png
echo.
echo  To stream the image response:
echo    python backend\medgemma_client.py path\to\your_mri.png --stream
echo.
pause
