@echo off
REM ═══════════════════════════════════════════════════════════════
REM run.bat  —  F1 Hybrid AI  •  Windows setup ^& launch helper
REM
REM Usage (double-click OR from Command Prompt / PowerShell):
REM   run.bat              -> full setup + launch Streamlit
REM   run.bat train        -> setup + train model only
REM   run.bat predict      -> setup + run sample predictions
REM   run.bat test         -> setup + run test suite
REM ═══════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ================================================
echo   F1 HYBRID AI  --  Windows Setup ^& Launch
echo ================================================
echo.

REM ── 1. Find Python 3.9-3.13 ──────────────────────────────────
set PYTHON=
for %%P in (python3.12 python3.11 python3.10 python3.9 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%V in ('%%P --version 2^>^&1') do (
            set PYVER=%%V
        )
        set PYTHON=%%P
        goto :python_found
    )
)
echo ERROR: Python 3.9-3.13 not found.
echo Download from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during install.
pause
exit /b 1
:python_found
echo [OK] Found %PYTHON% (%PYVER%)

REM ── 2. Virtual environment ────────────────────────────────────
if not exist ".venv\" (
    echo [>>] Creating virtual environment...
    %PYTHON% -m venv .venv
    echo [OK] Created .venv
) else (
    echo [OK] .venv already exists
)

call .venv\Scripts\activate.bat

REM ── 3. Install dependencies ───────────────────────────────────
echo [>>] Installing dependencies (first run takes ~60s)...
python -m pip install --upgrade pip -q
python -m pip install -e . -q
echo [OK] Dependencies installed

REM ── 4. Verify XGBoost ─────────────────────────────────────────
python -c "import xgboost" >nul 2>&1
if errorlevel 1 (
    echo [WARN] XGBoost failed to import.
    echo        Try: pip install xgboost==2.0.3
    pause
    exit /b 1
)
echo [OK] XGBoost verified

REM ── 5. Train if no saved model ────────────────────────────────
if not exist "models\saved\xgboost_f1.pkl" (
    echo [>>] No trained model found. Training now (takes ~30s)...
    python train.py
    echo [OK] Model trained
) else (
    echo [OK] Trained model already exists
)

REM ── 6. Action ─────────────────────────────────────────────────
set ACTION=%1
if "%ACTION%"=="" set ACTION=streamlit

if "%ACTION%"=="train" (
    python train.py
    goto :done
)
if "%ACTION%"=="predict" (
    echo.
    echo Monaco 2024 Dry:
    python predict.py --circuit monaco --year 2024
    echo.
    echo Spa 1988 Wet:
    python predict.py --circuit spa --year 1988 --rainfall 20
    goto :done
)
if "%ACTION%"=="test" (
    python test_model.py
    goto :done
)
if "%ACTION%"=="analysis" (
    python analysis.py
    goto :done
)
if "%ACTION%"=="streamlit" (
    echo.
    echo  Opening dashboard at http://localhost:8501
    echo  Press Ctrl+C to stop.
    echo.
    streamlit run app.py
    goto :done
)

echo Unknown action: %ACTION%
echo Usage: run.bat [streamlit^|train^|predict^|test^|analysis]
:done
endlocal
