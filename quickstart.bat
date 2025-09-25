@echo off
REM Author: Rohit Gangupantulu
REM Creative Automation Platform - Quick Start Script

echo ========================================
echo Creative Automation Platform - Quick Start
echo Author: Rohit Gangupantulu
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Warning: Some dependencies may have failed to install.
    echo You may need to install them manually.
)

echo.
echo Creating necessary directories...
if not exist outputs mkdir outputs
if not exist outputs\campaigns mkdir outputs\campaigns
if not exist outputs\assets mkdir outputs\assets
if not exist outputs\logs mkdir outputs\logs
if not exist campaign_briefs mkdir campaign_briefs
if not exist temp mkdir temp
if not exist examples mkdir examples

echo.
echo Setting up environment...
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo Created .env file from template
        echo Please edit .env file with your API keys (optional)
    ) else (
        echo Warning: .env.example not found
    )
) else (
    echo .env file already exists
)

echo.
echo ========================================
echo Setup complete! 
echo.
echo To run the platform:
echo   1. Web Interface: streamlit run app.py
echo   2. Command Line: python src/main.py demo
echo   3. Process campaign: python src/main.py process examples/sample_simple.json
echo   4. Run tests: pytest tests/ -v
echo.
echo Note: API keys (OpenAI or Stability AI) are required for image generation
echo ========================================
pause
