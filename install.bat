@echo off
REM Installation script for Person Detection System
REM This script installs all required Python packages

echo ========================================
echo Person Detection System - Installation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.10 from https://www.python.org/
        echo NOTE: torchreid only supports Python 3.10
        pause
        exit /b 1
    )
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo [1/4] Checking Python installation...
%PYTHON_CMD% --version
echo.
echo IMPORTANT: torchreid only supports Python 3.10
echo If you have a different version, please install Python 3.10
echo.

REM Check if pip is installed
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please install pip or reinstall Python with pip included
    pause
    exit /b 1
)

echo [2/4] Checking pip installation...
%PYTHON_CMD% -m pip --version
echo.

REM Ask user if they want to create a virtual environment
set /p CREATE_VENV="Do you want to create a virtual environment? (recommended) [Y/n]: "
if /i "%CREATE_VENV%"=="n" goto skip_venv
if /i "%CREATE_VENV%"=="no" goto skip_venv

echo [3/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

:skip_venv
echo [4/4] Installing required packages...
echo This may take several minutes...
echo.

%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo Verifying Streamlit installation...
%PYTHON_CMD% -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Streamlit verification failed, but installation may have succeeded
    echo You can verify manually by running: streamlit --version
) else (
    echo Streamlit installed successfully!
)
echo.

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Configure MongoDB connection (see README.md)
echo 2. Download model files to the model/ directory
echo 3. Run the application:
echo    - CLI: %PYTHON_CMD% py/main.py --sources 0 --track --reid --cloth-detect
echo    - Web: %PYTHON_CMD% -m streamlit run frontend/app.py
echo.
if defined VIRTUAL_ENV (
    echo Note: Virtual environment is active. Run 'deactivate' to exit.
) else (
    echo Note: Consider using a virtual environment for better isolation.
)
echo.
pause

