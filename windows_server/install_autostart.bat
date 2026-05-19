@echo off
echo Installing ForexBot MT5 Bridge as a startup task...

:: Get the directory where this script lives
set SCRIPT_DIR=%~dp0
set BRIDGE_BAT=%SCRIPT_DIR%start_bridge.bat

:: Create scheduled task that runs at logon for any user
schtasks /create /tn "ForexBot MT5 Bridge" /tr "\"%BRIDGE_BAT%\"" /sc onlogon /ru "%USERNAME%" /rl highest /f

if errorlevel 1 (
    echo.
    echo Scheduled task failed. Trying startup folder instead...
    copy "%BRIDGE_BAT%" "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\ForexBot_MT5_Bridge.bat"
    echo Installed to Startup folder.
) else (
    echo.
    echo Success! "ForexBot MT5 Bridge" will now start automatically at login.
)

echo.
echo Starting bridge now...
start "ForexBot MT5 Bridge" "%BRIDGE_BAT%"
echo Bridge started. Keep the new window open while trading.
pause
