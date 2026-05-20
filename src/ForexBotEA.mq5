//+------------------------------------------------------------------+
//| ForexBotEA.mq5 — File-based bridge for Linux forex bot          |
//| Reads commands from forexbot_cmd.txt, writes results to         |
//| forexbot_res.txt in the MQL5/Files folder.                      |
//|                                                                  |
//| Command format:                                                  |
//|   INFO                         → account info                   |
//|   TICK|SYMBOL                  → bid/ask                        |
//|   BUY|SYMBOL|VOL|SL|TP        → market buy                     |
//|   SELL|SYMBOL|VOL|SL|TP       → market sell                    |
//|   CLOSE|TICKET                 → close position                 |
//|   POSITIONS                    → list open positions            |
//+------------------------------------------------------------------+
#property copyright "ForexBot"
#property version   "1.00"
#property strict

input int PollMs = 100;   // Poll interval in milliseconds

datetime g_last_cmd_time = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   EventSetMillisecondTimer(PollMs);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTick() {}

//+------------------------------------------------------------------+
void OnTimer()
{
   string cmd_file = "forexbot_cmd.txt";
   string res_file = "forexbot_res.txt";

   if (!FileIsExist(cmd_file, FILE_COMMON)) return;

   int fh = FileOpen(cmd_file, FILE_READ | FILE_TXT | FILE_COMMON);
   if (fh == INVALID_HANDLE) return;

   string cmd = "";
   while (!FileIsEnding(fh))
      cmd += FileReadString(fh);
   FileClose(fh);
   StringTrimRight(cmd);
   StringTrimLeft(cmd);

   if (cmd == "") return;

   FileDelete(cmd_file, FILE_COMMON);

   string response = ProcessCommand(cmd);

   int rh = FileOpen(res_file, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if (rh != INVALID_HANDLE)
   {
      FileWriteString(rh, response);
      FileClose(rh);
   }
}

//+------------------------------------------------------------------+
string ProcessCommand(string cmd)
{
   string parts[];
   int n = StringSplit(cmd, '|', parts);
   if (n == 0) return "ERR|empty command";

   string op = parts[0];

   if (op == "INFO")   return CmdInfo();
   if (op == "TICK")   return (n >= 2) ? CmdTick(parts[1])   : "ERR|missing symbol";
   if (op == "BUY")    return (n >= 5) ? CmdOrder(ORDER_TYPE_BUY,  parts) : "ERR|bad BUY args";
   if (op == "SELL")   return (n >= 5) ? CmdOrder(ORDER_TYPE_SELL, parts) : "ERR|bad SELL args";
   if (op == "CLOSE")  return (n >= 2) ? CmdClose((long)StringToInteger(parts[1])) : "ERR|missing ticket";
   if (op == "POSITIONS") return CmdPositions();

   return "ERR|unknown command: " + op;
}

//+------------------------------------------------------------------+
string CmdInfo()
{
   long   login        = AccountInfoInteger(ACCOUNT_LOGIN);
   double balance      = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity       = AccountInfoDouble(ACCOUNT_EQUITY);
   string server       = AccountInfoString(ACCOUNT_SERVER);
   int    term_trade   = (int)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   int    acct_expert  = (int)AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   int    acct_trade   = (int)AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   int    mql_trade    = (int)MQLInfoInteger(MQL_TRADE_ALLOWED);
   return StringFormat("OK|LOGIN=%d|BALANCE=%.2f|EQUITY=%.2f|SERVER=%s|TERM_AT=%d|ACCT_EXPERT=%d|ACCT_TRADE=%d|MQL_AT=%d",
                       login, balance, equity, server, term_trade, acct_expert, acct_trade, mql_trade);
}

//+------------------------------------------------------------------+
string CmdTick(string symbol)
{
   MqlTick tick;
   if (!SymbolInfoTick(symbol, tick))
      return "ERR|no tick for " + symbol;
   return StringFormat("OK|BID=%.5f|ASK=%.5f", tick.bid, tick.ask);
}

//+------------------------------------------------------------------+
string CmdOrder(ENUM_ORDER_TYPE type, string &parts[])
{
   if (!MQLInfoInteger(MQL_TRADE_ALLOWED))
      return "ERR|EA live trading not enabled — right-click EA on chart → Properties → Common → Allow live trading";
   if (!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      return "ERR|Terminal algo trading disabled — click Algo Trading button in toolbar";

   string symbol = parts[1];
   double volume = StringToDouble(parts[2]);
   double sl     = StringToDouble(parts[3]);
   double tp     = StringToDouble(parts[4]);

   MqlTradeRequest req  = {};
   MqlTradeResult  res  = {};

   MqlTick tick;
   if (!SymbolInfoTick(symbol, tick))
      return "ERR|no tick for " + symbol;

   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = symbol;
   req.volume       = volume;
   req.type         = type;
   req.price        = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
   req.sl           = sl;
   req.tp           = tp;
   req.deviation    = 20;
   req.magic        = 20250518;
   req.comment      = "ForexBot";
   req.type_time    = ORDER_TIME_GTC;
   req.type_filling = ORDER_FILLING_FOK;

   if (!OrderSend(req, res))
      return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));

   if (res.retcode == TRADE_RETCODE_DONE)
      return StringFormat("OK|TICKET=%d|PRICE=%.5f", res.order, res.price);

   return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));
}

//+------------------------------------------------------------------+
string CmdClose(long ticket)
{
   // PositionSelectByTicket() is unreliable in OnTimer context;
   // iterate by index (same approach as CmdPositions) to select the position.
   bool found = false;
   int  total = PositionsTotal();
   for (int i = 0; i < total; i++)
   {
      if ((long)PositionGetTicket(i) == ticket) { found = true; break; }
   }
   if (!found)
      return "ERR|position not found: " + IntegerToString(ticket);

   string symbol  = PositionGetString(POSITION_SYMBOL);
   double volume  = PositionGetDouble(POSITION_VOLUME);
   long   postype = PositionGetInteger(POSITION_TYPE);

   MqlTick tick;
   if (!SymbolInfoTick(symbol, tick))
      return "ERR|no tick for " + symbol;

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};

   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = symbol;
   req.volume       = volume;
   req.type         = (postype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.price        = (postype == POSITION_TYPE_BUY) ? tick.bid : tick.ask;
   req.position     = ticket;
   req.deviation    = 20;
   req.magic        = 20250518;
   req.comment      = "ForexBot close";
   req.type_time    = ORDER_TIME_GTC;
   req.type_filling = ORDER_FILLING_FOK;

   if (!OrderSend(req, res))
      return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));

   if (res.retcode == TRADE_RETCODE_DONE)
      return "OK|closed";

   return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));
}

//+------------------------------------------------------------------+
string CmdPositions()
{
   int total = PositionsTotal();
   if (total == 0) return "OK|NONE";

   string result = "OK";
   for (int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if (!PositionSelectByTicket(ticket)) continue;

      string sym    = PositionGetString(POSITION_SYMBOL);
      long   ptype  = PositionGetInteger(POSITION_TYPE);
      double vol    = PositionGetDouble(POSITION_VOLUME);
      double open   = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl     = PositionGetDouble(POSITION_SL);
      double tp     = PositionGetDouble(POSITION_TP);
      double profit = PositionGetDouble(POSITION_PROFIT);
      string dir    = (ptype == POSITION_TYPE_BUY) ? "BUY" : "SELL";

      result += StringFormat("|%d,%s,%s,%.2f,%.5f,%.5f,%.5f,%.2f",
                             ticket, sym, dir, vol, open, sl, tp, profit);
   }
   return result;
}

//+------------------------------------------------------------------+
string ResultRetcodeDescription(uint code)
{
   switch(code)
   {
      case TRADE_RETCODE_REQUOTE:           return "Requote";
      case TRADE_RETCODE_REJECT:            return "Rejected";
      case TRADE_RETCODE_CANCEL:            return "Cancelled";
      case TRADE_RETCODE_PLACED:            return "Placed";
      case TRADE_RETCODE_DONE:              return "Done";
      case TRADE_RETCODE_DONE_PARTIAL:      return "Partial";
      case TRADE_RETCODE_ERROR:             return "Error";
      case TRADE_RETCODE_TIMEOUT:           return "Timeout";
      case TRADE_RETCODE_INVALID:           return "Invalid";
      case TRADE_RETCODE_INVALID_VOLUME:    return "Invalid volume";
      case TRADE_RETCODE_INVALID_PRICE:     return "Invalid price";
      case TRADE_RETCODE_INVALID_STOPS:     return "Invalid stops";
      case TRADE_RETCODE_TRADE_DISABLED:    return "Trade disabled";
      case TRADE_RETCODE_MARKET_CLOSED:     return "Market closed";
      case TRADE_RETCODE_NO_MONEY:          return "No money";
      case TRADE_RETCODE_PRICE_CHANGED:     return "Price changed";
      case TRADE_RETCODE_PRICE_OFF:         return "Price off";
      case TRADE_RETCODE_INVALID_EXPIRATION:return "Invalid expiry";
      case TRADE_RETCODE_ORDER_CHANGED:     return "Order changed";
      case TRADE_RETCODE_TOO_MANY_REQUESTS: return "Too many requests";
      case TRADE_RETCODE_NO_CHANGES:        return "No changes";
      case TRADE_RETCODE_SERVER_DISABLES_AT:return "Server disabled AT";
      case TRADE_RETCODE_CLIENT_DISABLES_AT:return "Client disabled AT";
      case TRADE_RETCODE_LOCKED:            return "Locked";
      case TRADE_RETCODE_FROZEN:            return "Frozen";
      case TRADE_RETCODE_INVALID_FILL:      return "Invalid fill";
      case TRADE_RETCODE_CONNECTION:        return "No connection";
      case TRADE_RETCODE_ONLY_REAL:         return "Only real";
      case TRADE_RETCODE_LIMIT_ORDERS:      return "Limit orders";
      case TRADE_RETCODE_LIMIT_VOLUME:      return "Limit volume";
      case TRADE_RETCODE_INVALID_ORDER:     return "Invalid order";
      case TRADE_RETCODE_POSITION_CLOSED:   return "Position closed";
      default: return "Code " + IntegerToString(code);
   }
}
