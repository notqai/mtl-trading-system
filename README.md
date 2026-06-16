# MTL Trading System

Automated multi-indicator trading framework for XAUUSD (gold) scalping. Final Year Project (FYP2), Curtin University Malaysia — Mohammad Qaiser Siew (supervised by Mr. Pang Po Ken).

A signal fired by a TradingView Pine indicator is filtered through a local machine-learning relay before it is allowed to place a trade on MetaTrader 5. Each trade's outcome is reported back to the relay, which retrains the models online.

## Pipeline

```
TradingView (Pine v6 "MTL Zone" indicator)
        │  alert() fires on signal
        ▼
Cloudflare Tunnel (cloudflared) — public URL → localhost:5000
        │
        ▼
Flask relay (mtl_relay.py) — ML filtering (RF + SGD live, SDAGD shadow)
        │  if passed
        ▼
PineConnector (webhook.pineconnector.com)
        │
        ▼
MT5 terminal → PineConnector EA → places trade on broker (HFM)
        │
        ▼
MTL_ResultReporter EA — reports win/loss back to /result
        │
        ▼
Relay retrains RF / updates SGD & SDAGD with the labeled outcome
```

Five zone-detection sources feed signals, each tagged with a `zone=` and confluence count `conf=`:
FVG (Fair Value Gap), OB (Order Block), QZone (swing-based), MZ (Market Zone / engulfing), Base Zone (consolidation breakout).

## Machine learning layer

| Model | Role | Update | Status |
|-------|------|--------|--------|
| RandomForest | 60% of ensemble | batch retrain every 5 labeled trades (15-trade min) | live |
| SGD | 40% of ensemble | `partial_fit` after every labeled trade | live |
| SDAGD (`sdagd_classifier.py`) | custom Stochastic Diagonal Approximate Greatest Descent | online, every result | **shadow only** — does not affect live decisions |

The live ensemble filters a signal by predicting its win probability against a dynamic win-rate threshold (0.48–0.62). See `PROJECT_STATUS.md` for the honest evaluation of model performance and the univariate findings (session and risk:reward buckets currently carry more signal than the models exploit).

## Files

| File | Description |
|------|-------------|
| `mtl_relay.py` | Flask relay — ML filtering, logging, online retraining |
| `sdagd_classifier.py` | Standalone SDAGD optimizer, imported by the relay |
| `MTL_Zone.pine` | TradingView Pine v6 indicator (all five zone sources, HTF bias, martingale layering, PineConnector alerts) |
| `MTL_ResultReporter.mq5` / `.ex5` | Custom MT5 EA — reports fills/closes back to the relay |
| `PineConnector-MT5-EA-v3.54.3.ex5` | Third-party PineConnector bridge EA |
| `trade_log.json` | Accumulated signal/trade history (back up before reset) |
| `PROJECT_STATUS.md` | Full context handoff: infra state, resolved bugs, ML results, open questions |

## Setup

```bash
pip install -r requirements.txt
python mtl_relay.py            # starts the relay on localhost:5000
```

In a second terminal, start the tunnel and point the TradingView alert webhook at `<tunnel-url>/webhook`:

```bash
cloudflared tunnel --url http://localhost:5000
```

> Every `cloudflared` restart generates a new random URL. The TradingView alert webhook must be updated each time or signals silently fail to arrive.

### MT5 side

1. Attach the PineConnector EA and `MTL_ResultReporter` EA to an XAUUSD chart.
2. Tools → Options → Expert Advisors → allow WebRequest for `http://127.0.0.1:5000`.
3. Broker: HFM, symbol `XAUUSD` (no suffix).

## Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/webhook` | receive a signal from TradingView |
| POST | `/result` | receive a trade result from MT5 |
| GET  | `/match` | match a fill to a pending signal |
| GET  | `/model` | model state (incl. SDAGD phase/radius/shadow accuracy) |
| GET  | `/stats` | per-zone win/loss stats |
| GET  | `/log` | full trade log |
| GET  | `/params` | current adaptive parameters |
| POST | `/reset` | reset parameters |
| GET  | `/health` | health check |

---

*Educational / research project. Not financial advice.*
