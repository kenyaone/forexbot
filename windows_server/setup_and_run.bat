@echo off
echo ============================================
echo  ForexBot MT5 Bridge Server Setup
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo Installing required packages...
pip install MetaTrader5 rpyc plumbum

echo.
echo Starting bridge server on port 18812...
echo Linux bot will connect to this machine on port 18812
echo Keep this window open while trading.
echo Press Ctrl+C to stop.
echo.

python -m mt5linux --host 0.0.0.0 -p 18812
pause
