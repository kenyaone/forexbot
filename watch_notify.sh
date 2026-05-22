#!/bin/bash
# Background log watcher — sends Telegram alerts for key bot events

TOKEN="8930726757:AAG5QyxFgw2Gm2OXR5lIuS38mp8aTvetnNI"
CHAT="8586751863"
LOG="/home/tele/forex-bot/logs/bot.log"

send() {
  curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    -d chat_id="$CHAT" \
    -d parse_mode=HTML \
    -d text="$1" > /dev/null
}

tail -n 0 -f "$LOG" | while read -r line; do
  if echo "$line" | grep -q "Order placed:"; then
    send "✅ <b>Trade Opened</b>
$(echo "$line" | grep -o 'Order placed:.*')"

  elif echo "$line" | grep -q "CLOSED\|closed.*profit\|closed.*loss\|Trade closed"; then
    send "🔴 <b>Trade Closed</b>
$line"

  elif echo "$line" | grep -q "trailing stop\|Trailing stop"; then
    send "🔔 <b>Trailing Stop Moved</b>
$line"

  elif echo "$line" | grep -q "daily loss limit\|weekly loss\|HALTED\|PAUSED"; then
    send "⚠️ <b>Risk Alert</b>
$line"

  elif echo "$line" | grep -q "ERROR\|Exception\|Traceback"; then
    send "❌ <b>Bot Error</b>
$line"

  elif echo "$line" | grep -q "Cycle complete"; then
    equity=$(echo "$line" | grep -o 'Equity: [^ ]*' | head -1)
    pnl=$(echo "$line" | grep -o 'Daily P&L: [^ ]*' | head -1)
    send "📊 <b>Cycle complete</b> — $equity | $pnl"
  fi
done
