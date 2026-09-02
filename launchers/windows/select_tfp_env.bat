@echo off
setlocal EnableExtensions

rem ================================================================
rem TFP Conda environment selector
rem Writes the selected environment to config\conda_env.txt.
rem The launcher and TFP Studio read the same file automatically.
rem ================================================================
set "TFP_ROOT=%~dp0..\.."
set "TFP_ENV_FILE=%TFP_ROOT%\config\conda_env.txt"
cd /d "%TFP_ROOT%"

where conda >nul 2>nul
if errorlevel 1 (
    echo [TFP] Conda was not found on PATH.
    echo Open Anaconda Prompt once, or add Conda to PATH, then retry.
    pause
    exit /b 1
)

echo.
echo ================= Available Conda Environments =================
call conda env list
echo ================================================================
echo.
set /p "TFP_CONDA_ENV=Enter the Conda environment name for TFP: "
if "%TFP_CONDA_ENV%"=="" (
    echo [TFP] No environment selected. Nothing changed.
    pause
    exit /b 1
)

echo [TFP] Checking environment: %TFP_CONDA_ENV%
call conda run -n "%TFP_CONDA_ENV%" python -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo [TFP] Conda environment "%TFP_CONDA_ENV%" was not found or cannot run Python.
    echo Create it first, then run this selector again.
    pause
    exit /b 1
)

> "%TFP_ENV_FILE%" echo %TFP_CONDA_ENV%
echo.
echo [TFP] Selected environment saved: %TFP_CONDA_ENV%
echo [TFP] Config file: %TFP_ENV_FILE%
echo The next launch_tfp.bat run will use this environment automatically.
pause
endlocal
