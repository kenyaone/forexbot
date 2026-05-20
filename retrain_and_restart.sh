#!/bin/bash
cd /home/tele/forex-bot
/home/tele/forex-bot/venv/bin/python retrain_model.py >> /home/tele/forex-bot/logs/retrain.log 2>&1
pkill -f 'python.*src.main'
sleep 5
nohup /home/tele/forex-bot/venv/bin/python -m src.main >> /home/tele/forex-bot/logs/bot.log 2>&1 &
echo "$(date -u) retrain complete, bot restarted PID $!" >> /home/tele/forex-bot/logs/retrain.log
