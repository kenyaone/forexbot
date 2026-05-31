#!/bin/bash
cd /home/tele/forex-bot
/home/tele/forex-bot/venv/bin/python retrain_model.py >> /home/tele/forex-bot/logs/retrain.log 2>&1
systemctl restart forexbot.service
echo "$(date -u) retrain complete, bot restarted via systemd" >> /home/tele/forex-bot/logs/retrain.log
