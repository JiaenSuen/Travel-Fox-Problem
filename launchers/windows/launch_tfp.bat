@echo off
setlocal

rem ================================================================
rem TFP Windows launcher source
rem Environment is selected by config\conda_env.txt.
rem Run select_tfp_env.bat to change it without editing this file.
rem ================================================================
set "TFP_ROOT=%~dp0..\.."
set "TFP_ENV_FILE=%TFP_ROOT%\config\conda_env.txt"
set "TFP_CONDA_ENV=tfp-rl"
cd /d "%TFP_ROOT%"

if exist "%TFP_ENV_FILE%" (
    set /p TFP_CONDA_ENV=<"%TFP_ENV_FILE%"
)
if "%TFP_CONDA_ENV%"=="" set "TFP_CONDA_ENV=tfp-rl"

where conda >nul 2>nul
if errorlevel 1 (
    echo [TFP] Conda was not found on PATH.
    echo Open Anaconda Prompt once, or add Conda to PATH, then retry.
    pause
    exit /b 1
)

echo [TFP] Launching TFP Studio in Conda env: %TFP_CONDA_ENV%
call conda run --no-capture-output -n "%TFP_CONDA_ENV%" python tfp_studio.py
if errorlevel 1 (
    echo.
    echo [TFP] Launch failed for environment: %TFP_CONDA_ENV%
    echo Run select_tfp_env.bat to choose another environment.
    echo Or create/update the selected environment with setup_conda_env.bat.
    pause
    exit /b 1
)
endlocal
