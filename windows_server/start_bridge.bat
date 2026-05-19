@echo off
cd /d "%~dp0"
title ForexBot MT5 Bridge
python -m mt5linux --host 0.0.0.0 -p 18812
