@echo off
setlocal
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
    pause
    exit /b 1
)

echo [TFP] Preparing environment: %TFP_CONDA_ENV%
call conda run -n "%TFP_CONDA_ENV%" python -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [TFP] Environment does not exist. Creating it...
    call conda env create -n "%TFP_CONDA_ENV%" -f environment.yml
) else (
    echo [TFP] Environment exists. Updating it...
    call conda env update -n "%TFP_CONDA_ENV%" -f environment.yml --prune
)

if errorlevel 1 (
    echo [TFP] Environment setup failed.
    pause
    exit /b 1
)

echo.
echo [TFP] Environment ready: %TFP_CONDA_ENV%
pause
endlocal
