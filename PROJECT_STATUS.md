# FYP2 Trading Framework — Project Status

**Purpose of this file:** context handoff for Claude Code (or any future session) picking up this project. Read this before touching any code — it explains what exists, what's been fixed, what's still open, and why certain decisions were made.

**Owner:** Qai (Mohammad Qaiser Siew Bin Mohd Qhadaffie), Curtin University Malaysia, FYP2, supervised by Mr. Pang Po Ken.

---

## 1. What this system is

An automated multi-indicator trading framework for XAUUSD (gold) scalping:

```
TradingView (Pine Script v6 "MTL Zone" indicator)
        │  alert() fires on signal
        ▼
Cloudflare Tunnel (cloudflared) — public URL → localhost:5000
        │
        ▼
Flask relay (mtl_relay.py) — ML filtering (RF + SGD, SDAGD shadow) + logging
        │  if passed
        ▼
PineConnector (webhook.pineconnector.com)
        │
        ▼
MT5 terminal → PineConnector-MT5-EA-v3.54.3 → places trade on broker (HFM)
        │
        ▼
MTL_ResultReporter EA (v3) — reports win/loss back to the relay's /result endpoint
        │
        ▼
Relay retrains RF / updates SGD & SDAGD with the labeled outcome
```

Five zone-detection sources feed signals: FVG (Fair Value Gap), OB (Order Block), QZone (swing-based), MZ (Market Zone / engulfing), Base Zone (consolidation breakout). Each tagged signal carries a `zone=` and `conf=` (confluence count) field into the ML filter.

---

## 2. Current broker / infra state

- **Broker: HFM**, symbol `XAUUSD` (no suffix). **Do not use JM** — briefly tried switching, but every EA-originated order failed with MT5 error 4752 (ERR_TRADE_DISABLED) even though manual trades worked fine. Reverted to HFM.
- **Tunnel: Cloudflare Tunnel (`cloudflared.exe`)**, not ngrok. Ngrok's free tier was serving an interstitial warning page (`ERR_NGROK_6024`) to TradingView's automated webhook POSTs, silently eating every signal — confirmed via `Invoke-WebRequest` showing the raw interstitial HTML instead of the relay's JSON.
  - **Every `cloudflared` restart generates a brand-new random URL** (`https://<random-words>.trycloudflare.com`). TradingView's alert webhook URL must be updated to `<new-url>/webhook` every time the tunnel restarts, or signals silently fail to arrive (they won't even show as a delivery attempt in ngrok/Cloudflare's own request log — TradingView just can't reach the old dead URL).
  - `cloudflared.exe` currently lives on Desktop (`C:\Users\Admin\Desktop\cloudflared.exe`), copied there from its winget install path (`C:\Program Files (x86)\cloudflared\`).
- **Relay runs locally**: `python mtl_relay.py` in a separate terminal from the tunnel. Both must be running simultaneously.
- **PineConnector EA reactivation**: if the EA shows "To reactivate PineConnector EA, please send in an alert: '<license>,eaon,eaon'", send that payload directly to `https://webhook.pineconnector.com` via a manual TradingView alert or curl/PowerShell POST — this happens after inactivity or certain error states, unrelated to the trading logic itself.

---

## 3. Known resolved bugs (do not re-introduce)

1. **Symbol suffix hardcoding** — Pine alerts originally used `syminfo.ticker` instead of the `pcSymbol` input, so broker switches (HFM→JM→HFM) silently sent the wrong symbol string. Fixed: all four `alert()` calls (main buy/sell + 2 martingale layer alerts) now use `pcLicenseID + ",buy," + pcSymbol + ...` — always check `pcSymbol` input matches the current broker's exact symbol string when switching brokers.
2. **FVG retest-fire bug** — a new FVG zone's boundary (`fTop = low` of the creation candle) made `priceInGap` trivially true on its own creation bar, causing `wasIn` to become `true` one bar later without any genuine retest ever happening. **Fixed** by adding `bullFVGCreatedBar[]` / `bearFVGCreatedBar[]` tracking arrays and gating all InZone/fire logic behind `bar_index > createdBar`. This fix is applied in the current script — do not remove the `[FIX #3]`-tagged code.
3. **ML threshold accidentally left at 0.0** — during debugging, `get_threshold()` was temporarily hardcoded to `return 0.0` to bypass blocking while testing webhook delivery. **This was reverted** — the function is back to its dynamic win-rate-based logic (0.48–0.62 depending on recent 20-trade win rate). If you see `threshold: 0.0` in relay logs going forward, someone left the debug override in — revert it.
4. **Stale MT5 tick subscription** — root cause of a string of "Invalid Stops" (MT5 error 4756) rejections on 2026-09-03. The EA was computing SL/TP against a frozen XAUUSD quote (~4387) that had drifted ~85 points from the real live price (~4471), because the symbol's tick stream had silently stalled in MT5 (all trades correctly rejected by the broker, none filled). Fixed by re-subscribing the symbol in Market Watch (Hide → Show All) — no code or config bug involved. **If "Invalid Stops" recurs, check for a frozen quote first** (compare the EA's logged "XAUUSD now is X" against the live chart) before assuming it's an SL-distance/spread config issue.
5. **`_pcMinSL` gold floor** — currently set to `100.0` pips (was `50.0`). Raised as a general safety margin during the above debugging, though the actual root cause turned out to be the stale-tick issue, not SL distance. Left at 100 as a harmless buffer.

---

## 4. ML layer — current status and honest results

Three models exist in `mtl_relay.py`:
- **RF (RandomForestClassifier)** — batch retrain every 5 labeled trades after a 15-trade minimum. **Live**, 60% weight in ensemble.
- **SGD (SGDClassifier)** — `partial_fit` after every single labeled trade. **Live**, 40% weight in ensemble.
- **SDAGD (custom `sdagd_classifier.py`)** — an from-scratch implementation of Stochastic Diagonal Approximate Greatest Descent (B.S. Goh's AGD, adapted for single-sample online learning: diagonal Levenberg-Marquardt step with trust-region radius, switching to full Newton step once the search region contains the minimum). **Shadow mode only** — learns from every result via `update_sdagd()`, logged as `[SDAGD-SHADOW]`, exposed via `GET /model` (`sdagd_phase`, `sdagd_radius`, `sdagd_shadow_accuracy`), but **does not** influence `predict_win_prob()` / live blocking decisions.

**As of ~89 labeled trades (2026-08-31 trade_log.json):**
- RF/SGD live ensemble prediction accuracy: **50.0%** (coin-flip)
- SDAGD shadow accuracy: **51.1%**, and its high-confidence (>0.9 prob) predictions only achieved **44.1%** actual win rate — i.e. its confidence calibration is not informative
- Trivial "always predict loss" baseline would score **57.3%** (51 losses / 89 trades) — **both live models currently underperform this baseline**
- **Conclusion for supervisor reporting**: neither optimizer is the differentiator here. The real finding is that the 13-feature set doesn't yet carry enough signal at this sample size — see univariate breakdowns below, which show real structure the models aren't exploiting.

**Univariate findings (worth acting on before more optimizer work):**
- **Session** is the strongest single factor: London 56.4% WR (n=39) vs NY_PM 20.0% (n=10), Asian 26.7% (n=15), Off 0.0% (n=3)
- **RR ratio** 1.8–2.2 bucket: 66.7% WR (n=12) — best bucket by far. >3.0 RR: 14.3% WR (n=7) — worst.
- **Layer/martingale entries underperform initial entries**: 36.8% (n=38) vs 47.1% (n=51)
- Recommended next step direction: session-based filtering or RR-bucket-aware sizing, rather than continued optimizer (SGD vs SDAGD) experimentation, given the current sample size.

**Do not fold SDAGD into the live ensemble** based on current evidence — it has not demonstrated an advantage over RF/SGD.

---

## 5. Open question — HTF counter-trend risk sizing

Currently: `htfCounterRisk = 0.5` (half size) applies when a trade goes against the HTF EMA bias in Soft mode; with-trend trades get full `pcRisk` (1.0).

Data check (n=89 trade log): counter-trend (risk=0.5) trades won **42.0%** (n=50) vs with-trend (risk=1.0) at **43.6%** (n=39) — a gap of only 1.6 points, likely noise at this sample size. This calls into question whether halving counter-trend size is justified. **Under discussion** — considering testing `htfCounterRisk = 1.0` (equal sizing) for a stretch and re-comparing win rates once more data accumulates. Not yet decided/implemented.

---

## 6. Files in this project

- `MTL_Zone.pine` — the TradingView indicator (Pine Script v6). Contains all five zone-detection sources, HTF bias filter, martingale layering, breakeven/partial-close, and the PineConnector alert-building logic.
- `mtl_relay.py` — Flask relay (v5 + SDAGD shadow addition). Endpoints: `/webhook`, `/match`, `/result`, `/model`, `/stats`, `/log`, `/params`, `/reset`, `/health`.
- `sdagd_classifier.py` — standalone SDAGD implementation, imported by the relay.
- `trade_log.json` — accumulated signal/trade history (grows over time; back up before `/reset`).
- MT5 side: `PineConnector-MT5-EA-v3.54.3` (third-party, not ours to edit) + `MTL_ResultReporter` EA v3 (custom, reports fills/closes back to `/result` and `/match`).

---

## 7. What NOT to do without discussion

- Don't re-enable JM broker without checking whether HFM has become unavailable — JM has a confirmed, unresolved EA-trading restriction.
- Don't fold SDAGD into live trading decisions — shadow-only until it shows real out-of-sample skill.
- Don't leave `get_threshold()` hardcoded during future debugging without reverting it before resuming live signals.
- Don't assume an "Invalid Stops" error is an SL-distance/config problem without first checking for a stale MT5 tick feed (compare logged price vs live chart).
