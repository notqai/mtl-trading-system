//+------------------------------------------------------------------+
//| MTL_ResultReporter.mq5  v3.0                                     |
//| Runs alongside PineConnector EA.                                 |
//|                                                                  |
//| Fixes over v2.0:                                                 |
//|  1. Broker server time converted to UTC before matching.         |
//|  2. Matching moved server-side to the relay /match endpoint.     |
//|     No more fragile JSON parsing in MQL5.                        |
//|  3. Matches on position OPEN time, not close time.               |
//|  4. Partial closes aggregated. One position reports once.        |
//|  5. Never posts an empty trade_id. Unmatched trades are skipped  |
//|     and logged instead of corrupting the oldest pending entry.   |
//|                                                                  |
//| Setup: Tools > Options > Expert Advisors > Allow WebRequest      |
//|        Add: http://127.0.0.1:5000                                |
//+------------------------------------------------------------------+
#property copyright "MTL Zone"
#property strict
#property version   "3.0"

input string RelayResultURL  = "http://127.0.0.1:5000/result";
input string RelayMatchURL   = "http://127.0.0.1:5000/match";
input string TradeSymbol     = "XAUUSD";   // "" = all symbols
input int    PollSeconds     = 5;
input int    MatchWindowSec  = 180;        // max gap between signal and fill
input int    BrokerGMTOffset = 3;          // broker server hours ahead of UTC
input int    HistoryLookbackDays = 3;
input bool   DebugLog        = true;

ulong reported[];      // position ids already sent
datetime startupTime = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   startupTime = TimeCurrent();
   ArrayResize(reported, 0);

   // Seed with everything already closed so a restart does not resend.
   SeedReported();

   Print("[Reporter v3] Started. Match=", RelayMatchURL, " Result=", RelayResultURL);
   Print("[Reporter v3] Broker offset from UTC: ", BrokerGMTOffset, "h");
   Print("[Reporter v3] Seeded ", ArraySize(reported), " already-closed positions.");

   EventSetTimer(PollSeconds);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("[Reporter v3] Stopped.");
}

//+------------------------------------------------------------------+
//| Mark every already-closed position as reported at startup        |
//+------------------------------------------------------------------+
void SeedReported()
{
   datetime from = TimeCurrent() - (HistoryLookbackDays * 86400);
   if(!HistorySelect(from, TimeCurrent())) return;

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

      ulong posId = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      if(posId == 0) continue;
      MarkReported(posId);
   }
}

bool AlreadyReported(ulong posId)
{
   for(int i = 0; i < ArraySize(reported); i++)
      if(reported[i] == posId) return true;
   return false;
}

void MarkReported(ulong posId)
{
   if(AlreadyReported(posId)) return;
   int n = ArraySize(reported);
   ArrayResize(reported, n + 1);
   reported[n] = posId;
}

//+------------------------------------------------------------------+
void OnTimer()
{
   datetime from = TimeCurrent() - (HistoryLookbackDays * 86400);
   if(!HistorySelect(from, TimeCurrent())) return;

   // Pass 1: collect candidate position ids.
   // HistorySelectByPosition resets the selection, so we cannot call it
   // inside this loop.
   ulong candidates[];
   ArrayResize(candidates, 0);

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(TradeSymbol != "" && sym != TradeSymbol) continue;

      ulong posId = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      if(posId == 0) continue;
      if(AlreadyReported(posId)) continue;

      bool dup = false;
      for(int k = 0; k < ArraySize(candidates); k++)
         if(candidates[k] == posId) { dup = true; break; }
      if(dup) continue;

      int n = ArraySize(candidates);
      ArrayResize(candidates, n + 1);
      candidates[n] = posId;
   }

   // Pass 2: process each candidate position in full.
   for(int c = 0; c < ArraySize(candidates); c++)
      ProcessPosition(candidates[c]);
}

//+------------------------------------------------------------------+
//| Aggregate all deals of one position. Report only when fully      |
//| closed so partial closes produce a single result.                |
//+------------------------------------------------------------------+
void ProcessPosition(ulong posId)
{
   if(!HistorySelectByPosition(posId)) return;

   int total = HistoryDealsTotal();
   if(total == 0) return;

   double volIn = 0.0, volOut = 0.0, pnl = 0.0;
   datetime openTime = 0;
   string sym = "";

   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      long entry  = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      double vol  = HistoryDealGetDouble(ticket, DEAL_VOLUME);

      if(entry == DEAL_ENTRY_IN)
      {
         volIn += vol;
         datetime t = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
         if(openTime == 0 || t < openTime) openTime = t;
         if(sym == "") sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      }
      else if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT)
      {
         volOut += vol;
         pnl += HistoryDealGetDouble(ticket, DEAL_PROFIT)
              + HistoryDealGetDouble(ticket, DEAL_SWAP)
              + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         if(sym == "") sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      }
   }

   if(volIn <= 0.0) return;

   // Still partially open. Wait for the remainder.
   if(volOut < volIn - 0.0000001)
   {
      if(DebugLog)
         Print("[Reporter v3] Position ", posId, " partially closed (",
               DoubleToString(volOut,2), "/", DoubleToString(volIn,2), "). Waiting.");
      return;
   }

   string result = (pnl >= 0.0) ? "win" : "loss";

   // Broker server time to UTC. This is the fix that makes matching work.
   long openUTC = (long)openTime - ((long)BrokerGMTOffset * 3600);

   if(DebugLog)
      Print("[Reporter v3] Position ", posId, " closed | ", result,
            " | pnl=", DoubleToString(pnl,2),
            " | openUTC=", (string)openUTC);

   string tradeId = "";
   string zone    = "UNKNOWN";

   if(!FetchMatch(openUTC, tradeId, zone))
   {
      Print("[Reporter v3] NO MATCH for position ", posId,
            ". Not posting. Check BrokerGMTOffset and MatchWindowSec.");
      MarkReported(posId);   // avoid retry spam
      return;
   }

   if(PostResult(tradeId, zone, result, pnl, sym))
      MarkReported(posId);
}

//+------------------------------------------------------------------+
//| Ask the relay which pending signal matches this fill time        |
//+------------------------------------------------------------------+
bool FetchMatch(long tsUTC, string &tradeId, string &zone)
{
   tradeId = "";
   zone    = "UNKNOWN";

   string url = RelayMatchURL + "?ts=" + (string)tsUTC
              + "&window=" + (string)MatchWindowSec;

   char   post[];
   char   response[];
   string responseHeaders;
   string headers = "";

   ResetLastError();
   int res = WebRequest("GET", url, headers, 5000, post, response, responseHeaders);

   if(res == -1)
   {
      Print("[Reporter v3] /match request failed. Error ", GetLastError());
      Print("[Reporter v3] Whitelist http://127.0.0.1:5000 in Tools > Options > Expert Advisors.");
      return false;
   }

   string json = CharArrayToString(response);
   if(DebugLog) Print("[Reporter v3] /match -> ", json);

   tradeId = JsonString(json, "id");
   zone    = JsonString(json, "zone");
   if(zone == "") zone = "UNKNOWN";

   return (tradeId != "" && tradeId != "null");
}

//+------------------------------------------------------------------+
//| Minimal helper: read "key":"value" out of a small flat JSON      |
//+------------------------------------------------------------------+
string JsonString(string json, string key)
{
   string needle = "\"" + key + "\"";
   int p = StringFind(json, needle);
   if(p < 0) return "";

   p += StringLen(needle);
   while(p < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, p);
      if(ch == ' ' || ch == ':') { p++; continue; }
      break;
   }
   if(p >= StringLen(json)) return "";
   if(StringGetCharacter(json, p) != '"') return "";   // null or numeric

   p++;
   int end = StringFind(json, "\"", p);
   if(end < 0) return "";
   return StringSubstr(json, p, end - p);
}

//+------------------------------------------------------------------+
bool PostResult(string tradeId, string zone, string result,
                double pnl, string symbol)
{
   if(tradeId == "")
   {
      Print("[Reporter v3] Refusing to post with empty trade_id.");
      return false;
   }

   string json = StringFormat(
      "{\"trade_id\":\"%s\",\"zone\":\"%s\",\"result\":\"%s\",\"pnl\":%.2f,\"symbol\":\"%s\"}",
      tradeId, zone, result, pnl, symbol);

   char   post[];
   char   response[];
   string responseHeaders;
   string headers = "Content-Type: application/json\r\n";

   int len = StringLen(json);
   StringToCharArray(json, post, 0, len);
   ArrayResize(post, len);   // drop trailing null byte

   ResetLastError();
   int res = WebRequest("POST", RelayResultURL, headers, 5000, post, response, responseHeaders);

   if(res == -1)
   {
      Print("[Reporter v3] POST failed. Error ", GetLastError());
      return false;
   }

   string body = CharArrayToString(response);
   if(DebugLog) Print("[Reporter v3] Relay response: ", body);

   // Relay returns 409 with an error body if the id was already labeled.
   if(res != 200)
   {
      Print("[Reporter v3] Relay rejected result. HTTP ", res, " ", body);
      return false;
   }

   return true;
}

void OnTick() {}
//+------------------------------------------------------------------+
