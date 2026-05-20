//+------------------------------------------------------------------+
//| ForexBotScript.mq5 — Script version of file bridge              |
//| Scripts always have MQL_TRADE_ALLOWED=1 when algo trading is ON |
//| Runs in infinite loop until MT5 closes or script is stopped.    |
//+------------------------------------------------------------------+
#property script_show_confirm false
#property script_show_inputs  false

//+------------------------------------------------------------------+
void OnStart()
{
   Print("ForexBotScript started — polling for commands");

   while (!IsStopped())
   {
      ProcessPendingCommand();
      Sleep(100);
   }

   Print("ForexBotScript stopped");
}

//+------------------------------------------------------------------+
void ProcessPendingCommand()
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

   if (op == "INFO")      return CmdInfo();
   if (op == "TICK")      return (n >= 2) ? CmdTick(parts[1])                  : "ERR|missing symbol";
   if (op == "BUY")       return (n >= 5) ? CmdOrder(ORDER_TYPE_BUY,  parts)   : "ERR|bad BUY args";
   if (op == "SELL")      return (n >= 5) ? CmdOrder(ORDER_TYPE_SELL, parts)   : "ERR|bad SELL args";
   if (op == "CLOSE")     return (n >= 2) ? CmdClose((long)StringToInteger(parts[1])) : "ERR|missing ticket";
   if (op == "POSITIONS") return CmdPositions();

   return "ERR|unknown command: " + op;
}

//+------------------------------------------------------------------+
string CmdInfo()
{
   long   login       = AccountInfoInteger(ACCOUNT_LOGIN);
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   string server      = AccountInfoString(ACCOUNT_SERVER);
   int    term_at     = (int)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   int    mql_at      = (int)MQLInfoInteger(MQL_TRADE_ALLOWED);
   int    acct_expert = (int)AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   return StringFormat("OK|LOGIN=%d|BALANCE=%.2f|EQUITY=%.2f|SERVER=%s|TERM_AT=%d|MQL_AT=%d|ACCT_EXPERT=%d",
                       login, balance, equity, server, term_at, mql_at, acct_expert);
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
   string symbol = parts[1];
   double volume = StringToDouble(parts[2]);
   double sl     = StringToDouble(parts[3]);
   double tp     = StringToDouble(parts[4]);

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
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
      return "OK|TICKET=" + (string)res.order + StringFormat("|PRICE=%.5f", res.price);

   return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));
}

//+------------------------------------------------------------------+
string CmdClose(long ticket)
{
   if (!PositionSelectByTicket(ticket))
      return "ERR|position not found: " + IntegerToString(ticket);

   string symbol  = PositionGetString(POSITION_SYMBOL);
   double volume  = PositionGetDouble(POSITION_VOLUME);
   long   postype = PositionGetInteger(POSITION_TYPE);

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   MqlTick tick;

   if (!SymbolInfoTick(symbol, tick))
      return "ERR|no tick for " + symbol;

   req.action       = TRADE_ACTION_DEAL;
   req.position     = ticket;
   req.symbol       = symbol;
   req.volume       = volume;
   req.type         = (postype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.price        = (postype == POSITION_TYPE_BUY) ? tick.bid : tick.ask;
   req.deviation    = 20;
   req.magic        = 20250518;
   req.type_filling = ORDER_FILLING_FOK;

   if (!OrderSend(req, res))
      return StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));

   return (res.retcode == TRADE_RETCODE_DONE) ? "OK|CLOSED" :
          StringFormat("ERR|%d: %s", res.retcode, ResultRetcodeDescription(res.retcode));
}

//+------------------------------------------------------------------+
string CmdPositions()
{
   int total = PositionsTotal();
   if (total == 0) return "OK|NONE";

   string result = "OK";
   for (int i = 0; i < total; i++)
   {
      ulong  ticket  = PositionGetTicket(i);
      string symbol  = PositionGetString(POSITION_SYMBOL);
      long   ptype   = PositionGetInteger(POSITION_TYPE);
      double volume  = PositionGetDouble(POSITION_VOLUME);
      double oprice  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl      = PositionGetDouble(POSITION_SL);
      double tp      = PositionGetDouble(POSITION_TP);
      double profit  = PositionGetDouble(POSITION_PROFIT);
      string dir     = (ptype == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      result += "|" + (string)ticket + StringFormat(",%s,%s,%.2f,%.5f,%.5f,%.5f,%.2f",
                             symbol, dir, volume, oprice, sl, tp, profit);
   }
   return result;
}
