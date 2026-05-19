import smtplib
import os
import logging
import urllib.request
import urllib.parse
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

_smtp_user    = os.getenv('ALERT_SMTP_USER', '')
_smtp_pass    = os.getenv('ALERT_SMTP_PASS', '')
_alert_email  = os.getenv('ALERT_EMAIL', '')
_tg_token     = os.getenv('TELEGRAM_BOT_TOKEN', '')
_tg_chat_id   = os.getenv('TELEGRAM_CHAT_ID', '')


def _send_telegram(text: str):
    if not all([_tg_token, _tg_chat_id]):
        return
    try:
        url  = f"https://api.telegram.org/bot{_tg_token}/sendMessage"
        data = urllib.parse.urlencode({
            'chat_id':    _tg_chat_id,
            'text':       text,
            'parse_mode': 'HTML',
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
        logger.info("Telegram alert sent")
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")


def _send_email(subject: str, body: str):
    if not all([_smtp_user, _smtp_pass, _alert_email]):
        return
    try:
        msg = MIMEMultipart()
        msg['From']    = _smtp_user
        msg['To']      = _alert_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(_smtp_user, _smtp_pass)
            s.send_message(msg)
        logger.info(f"Email alert sent: {subject}")
    except Exception as e:
        logger.warning(f"Email alert failed: {e}")


def _send(subject: str, body: str, tg_text: str = None):
    _send_email(subject, body)
    _send_telegram(tg_text or body)


def alert_trade_opened(pair: str, direction: str, volume: float,
                       entry: float, sl: float, tp: float,
                       confidence: float, ticket: int):
    pips_risk   = abs(entry - sl) * 10000
    pips_reward = abs(tp - entry) * 10000
    rr          = pips_reward / pips_risk if pips_risk else 0

    subject = f"ForexBot — {direction} {pair} opened"
    body = (
        f"Trade Opened\n"
        f"{'='*30}\n"
        f"Pair:       {pair}\n"
        f"Direction:  {direction}\n"
        f"Volume:     {volume} lots\n"
        f"Entry:      {entry:.5f}\n"
        f"Stop Loss:  {sl:.5f}  ({pips_risk:.1f} pips risk)\n"
        f"Take Profit:{tp:.5f}  ({pips_reward:.1f} pips reward)\n"
        f"R:R Ratio:  1:{rr:.1f}\n"
        f"ML Conf:    {confidence:.1f}%\n"
        f"Ticket:     {ticket}\n"
        f"Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )
    tg = (
        f"<b>🟢 TRADE OPENED</b>\n"
        f"<b>{direction} {pair}</b>\n"
        f"Entry: <code>{entry:.5f}</code>\n"
        f"SL: <code>{sl:.5f}</code> ({pips_risk:.1f} pips)\n"
        f"TP: <code>{tp:.5f}</code> ({pips_reward:.1f} pips)\n"
        f"R:R: 1:{rr:.1f} | Lot: {volume} | Conf: {confidence:.1f}%\n"
        f"Ticket: {ticket} | {datetime.utcnow().strftime('%H:%M UTC')}"
    )
    _send(subject, body, tg)


def alert_trade_closed(pair: str, direction: str, volume: float,
                       entry: float, close_price: float,
                       profit: float, ticket: int):
    result = 'PROFIT' if profit >= 0 else 'LOSS'
    pips   = (close_price - entry) * 10000 if direction == 'BUY' else (entry - close_price) * 10000
    emoji  = '💰' if profit >= 0 else '🔴'

    subject = f"ForexBot — {pair} closed | {result} ${profit:+.2f}"
    body = (
        f"Trade Closed\n"
        f"{'='*30}\n"
        f"Pair:       {pair}\n"
        f"Direction:  {direction}\n"
        f"Volume:     {volume} lots\n"
        f"Entry:      {entry:.5f}\n"
        f"Close:      {close_price:.5f}\n"
        f"Pips:       {pips:+.1f}\n"
        f"P&L:        ${profit:+.2f}\n"
        f"Ticket:     {ticket}\n"
        f"Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )
    tg = (
        f"<b>{emoji} TRADE CLOSED — {result}</b>\n"
        f"<b>{direction} {pair}</b>\n"
        f"Entry: <code>{entry:.5f}</code> → Close: <code>{close_price:.5f}</code>\n"
        f"Pips: {pips:+.1f} | P&L: <b>${profit:+.2f}</b>\n"
        f"Ticket: {ticket} | {datetime.utcnow().strftime('%H:%M UTC')}"
    )
    _send(subject, body, tg)


def alert_daily_summary(equity: float, balance: float, daily_pnl: float, trades: int):
    result = 'UP' if daily_pnl >= 0 else 'DOWN'
    emoji  = '📈' if daily_pnl >= 0 else '📉'

    subject = f"ForexBot Daily Summary — {result} ${daily_pnl:+.2f}"
    body = (
        f"Daily Summary\n"
        f"{'='*30}\n"
        f"Balance:    ${balance:.2f}\n"
        f"Equity:     ${equity:.2f}\n"
        f"Daily P&L:  ${daily_pnl:+.2f}\n"
        f"Trades:     {trades}\n"
        f"Date:       {datetime.utcnow().strftime('%Y-%m-%d UTC')}\n"
    )
    tg = (
        f"<b>{emoji} DAILY SUMMARY</b>\n"
        f"Balance: ${balance:.2f} | Equity: ${equity:.2f}\n"
        f"P&L: <b>${daily_pnl:+.2f}</b> | Trades: {trades}\n"
        f"{datetime.utcnow().strftime('%Y-%m-%d UTC')}"
    )
    _send(subject, body, tg)


def alert_error(message: str):
    subject = "ForexBot — Error"
    body    = f"Error at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}:\n\n{message}"
    tg      = f"<b>⚠️ FOREXBOT ERROR</b>\n<code>{message[:300]}</code>"
    _send(subject, body, tg)


def alert_warning(message: str):
    subject = "ForexBot — Warning"
    body    = f"Warning at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}:\n\n{message}"
    tg      = f"<b>⚠️ WARNING</b>\n{message}"
    _send(subject, body, tg)
