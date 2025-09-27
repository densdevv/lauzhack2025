@echo off
setlocal ENABLEDELAYEDEXPANSION

REM -----------------------------------------------------------------
REM Swiss Weather Intelligence System - Launcher (Windows)
REM Robust: works when double-clicked or run from any directory
REM -----------------------------------------------------------------

REM Move to the folder where this script resides
pushd "%~dp0" 1>nul 2>nul

echo ================================================================
echo Swiss Weather Intelligence System - LauzHack 2025
echo ================================================================
echo.
echo Starting Advanced Weather Intelligence Web Application...
echo.

REM Detect Python (prefer python, fallback to py -3)
set "PY_CMD="
where python 1>nul 2>nul && set "PY_CMD=python"
if not defined PY_CMD (
    where py 1>nul 2>nul && set "PY_CMD=py -3"
)
if not defined PY_CMD (
    echo [ERROR] Python not found. Please install Python 3.9+ and ensure it is on PATH.
    echo Download: https://www.python.org/downloads/
    popd
    exit /b 1
)

echo 🔧 Checking Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY_CMD% -m venv ".venv"
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"
set "VENV_STREAMLIT=.venv\Scripts\streamlit.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment creation failed. Path not found: %VENV_PY%
    echo Ensure you have permission to create files in this folder.
    popd
    exit /b 1
)

echo.
echo 📦 Installing required dependencies (requirements.txt)...
"%VENV_PY%" -m pip install --upgrade pip 1>nul
if exist "requirements.txt" (
    "%VENV_PIP%" install -r requirements.txt
)
REM Fallback: if requirements.txt is missing, install core libs directly
if %errorlevel% NEQ 0 (
    echo requirements.txt install failed or missing, installing core packages directly...
    "%VENV_PIP%" install streamlit pandas numpy plotly requests scipy scikit-learn matplotlib seaborn
)

echo.
echo 🌐 Starting Streamlit Web Application...
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ================================================================
echo.

if exist "%VENV_STREAMLIT%" (
    "%VENV_STREAMLIT%" run weather_app.py --server.headless false --server.port 8501 --server.address localhost
) else (
    "%VENV_PY%" -m streamlit run weather_app.py --server.headless false --server.port 8501 --server.address localhost
)

echo.
echo Application stopped. Press any key to exit...
pause

popd 1>nul 2>nul
endlocal