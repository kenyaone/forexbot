import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

_smtp_user = os.getenv('ALERT_SMTP_USER', '')
_smtp_pass = os.getenv('ALERT_SMTP_PASS', '')
_alert_email = os.getenv('ALERT_EMAIL', '')


def _send(subject: str, body: str):
    if not all([_smtp_user, _smtp_pass, _alert_email]):
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = _smtp_user
        msg['To'] = _alert_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(_smtp_user, _smtp_pass)
            s.send_message(msg)
        logger.info(f"Alert sent: {subject}")
    except Exception as e:
        logger.warning(f"Alert failed: {e}")


def alert_trade_opened(pair: str, direction: str, volume: float,
                       entry: float, sl: float, tp: float,
                       confidence: float, ticket: int):
    pips_risk = abs(entry - sl) * 10000
    pips_reward = abs(tp - entry) * 10000
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
        f"R:R Ratio:  1:{pips_reward/pips_risk:.1f}\n"
        f"ML Conf:    {confidence:.1f}%\n"
        f"Ticket:     {ticket}\n"
        f"Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )
    _send(subject, body)


def alert_trade_closed(pair: str, direction: str, volume: float,
                       entry: float, close_price: float,
                       profit: float, ticket: int):
    result = 'PROFIT' if profit >= 0 else 'LOSS'
    pips = (close_price - entry) * 10000 if direction == 'BUY' else (entry - close_price) * 10000
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
    _send(subject, body)


def alert_daily_summary(equity: float, balance: float, daily_pnl: float, trades: int):
    result = 'UP' if daily_pnl >= 0 else 'DOWN'
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
    _send(subject, body)


def alert_error(message: str):
    subject = "ForexBot — Error"
    body = f"Error at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}:\n\n{message}"
    _send(subject, body)
