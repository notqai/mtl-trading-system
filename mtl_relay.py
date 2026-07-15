"""
MTL Zone ML Relay v5 - Enhanced Random Forest + Online Learning

Changes from v4:
  - /match endpoint: the reporter EA asks the relay which pending signal
    corresponds to a fill time. All JSON work happens here in Python
    instead of being hand-parsed in MQL5.
  - /result no longer falls back to the oldest pending entry. An unmatched
    or already-labeled trade_id is rejected with 409. A dropped result is
    recoverable. A mislabeled one poisons the model permanently.
  - File paths anchored to the script directory so the working directory
    can no longer silently split the models and log into two sets.
  - Zone matching is case-insensitive. "Base" added to ZONE_CLASSES.
  - LICENSE_ID updated to 9355302928597.

NOTE: adding "Base" shifts the zone encoding in feature 0. Old rows use the
      previous indices. Run POST /reset before collecting new data.

Changes in this revision:
  - Added SDAGDClassifier (sdagd_classifier.py) as a SHADOW model. It
    receives the same partial_fit updates as SGD after every labeled
    trade, and its predictions/phase/radius are surfaced via /model, but
    it does NOT currently participate in predict_win_prob()'s blocking
    decision. This lets it accumulate a track record for comparison
    against RF/SGD before it's trusted in the live ensemble.
"""

from flask import Flask, request, jsonify
import time
import requests, json, os, logging, threading, random
import numpy as np
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sdagd_classifier import SDAGDClassifier

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))

PINECONNECTOR_URL   = "https://webhook.pineconnector.com"
LICENSE_ID          = "8832727941333"
LOG_FILE            = os.path.join(BASE_DIR, "trade_log.json")
RF_FILE             = os.path.join(BASE_DIR, "rf_model.joblib")
SGD_FILE            = os.path.join(BASE_DIR, "sgd_model.joblib")
SDAGD_FILE          = os.path.join(BASE_DIR, "sdagd_model.joblib")
SCALER_FILE         = os.path.join(BASE_DIR, "scaler.joblib")

RF_MIN_SAMPLES      = 15    # min labeled trades before RF activates
RF_RETRAIN_EVERY    = 5     # retrain RF every N new labeled trades
SGD_MIN_SAMPLES     = 2     # SGD needs at least 1 win + 1 loss seen
GOLD_MIN_SL_PIPS    = 50.0

MATCH_WINDOW_SEC    = 180   # default fill-to-signal window
MATCH_LEAD_SEC      = 30    # tolerance for a fill stamped before its signal

# ── Exploration ───────────────────────────────────────────────
# A filter trained on a mostly-losing sample will block everything, which
# stops new results arriving, which freezes the model permanently. Two guards:
#   1. No blocking at all until MIN_LABELED_TO_BLOCK results are in.
#   2. After that, EXPLORE_RATE of below-threshold signals pass anyway so the
#      feedback loop keeps receiving labels.
# Explored trades are tagged decision="explore" so they can be separated from
# model-approved trades during analysis.
MIN_LABELED_TO_BLOCK = 30
EXPLORE_RATE         = 0.20

ZONE_CLASSES  = ["FVG", "OB", "QZone", "MZ", "Base", "LAYER", "UNKNOWN"]
ZONE_MAP      = {z: i for i, z in enumerate(ZONE_CLASSES)}
ZONE_LOOKUP   = {z.lower(): z for z in ZONE_CLASSES}

# ── Feature names (13 total) ──────────────────────────────────
FEATURE_NAMES = [
    "zone",           # 0  encoded
    "confluence",     # 1  1-4
    "direction",      # 2  1=buy 0=sell
    "sl_pips",        # 3  float
    "tp_pips",        # 4  float
    "rr_ratio",       # 5  tp/sl
    "risk",           # 6  % risk
    "hour",           # 7  0-23 UTC
    "day_of_week",    # 8  0=Mon
    "session",        # 9  0=Asian 1=London 2=NY_AM 3=NY_PM 4=Off
    "volatility",     # 10 sl relative to rolling avg (1.0=normal)
    "rolling_wr",     # 11 last-10 win rate 0.0-1.0
    "loss_streak",    # 12 consecutive losses capped at 5
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log  = logging.getLogger(__name__)
lock = threading.Lock()


# ── Persistence ───────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f: return json.load(f)
    return []

def save_log(data):
    with open(LOG_FILE, "w") as f: json.dump(data, f, indent=2)

def load_models():
    rf    = joblib.load(RF_FILE)     if os.path.exists(RF_FILE)     else None
    sgd   = joblib.load(SGD_FILE)    if os.path.exists(SGD_FILE)    else None
    sdagd = joblib.load(SDAGD_FILE)  if os.path.exists(SDAGD_FILE)  else None
    sc    = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else StandardScaler()
    return rf, sgd, sdagd, sc

def save_models(rf, sgd, sdagd, sc):
    if rf    is not None: joblib.dump(rf,    RF_FILE)
    if sgd   is not None: joblib.dump(sgd,   SGD_FILE)
    if sdagd is not None: joblib.dump(sdagd, SDAGD_FILE)
    joblib.dump(sc, SCALER_FILE)


# ── Time helpers ──────────────────────────────────────────────
def utc_now_iso():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def iso_to_epoch(ts):
    """Parse the relay's own ISO timestamps into a UTC epoch int."""
    try:
        return int(datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None


def normalise_zone(raw):
    if not raw:
        return "UNKNOWN"
    return ZONE_LOOKUP.get(str(raw).strip().lower(), "UNKNOWN")


# ── Session helper ────────────────────────────────────────────
def get_session(hour):
    if  0 <= hour <  8: return 0   # Asian
    if  8 <= hour < 12: return 1   # London
    if 12 <= hour < 17: return 2   # NY AM
    if 17 <= hour < 21: return 3   # NY PM
    return 4                        # Off


# ── Dynamic threshold ─────────────────────────────────────────
def get_threshold(trade_log):
    # Filter to labeled trades FIRST, then take the last 20. Slicing the raw
    # log first breaks once blocked signals are persisted, because the tail
    # fills with blocks and no labeled trades remain in the window.
    labeled = [t for t in trade_log if t.get("result") in ("win", "loss")]
    recent = labeled[-20:]
    if len(recent) < 5:
        return 0.52
    wr = sum(1 for t in recent if t["result"] == "win") / len(recent)
    if   wr > 0.65: return 0.48
    elif wr > 0.55: return 0.52
    elif wr > 0.45: return 0.57
    else:           return 0.62


# ── Volatility regime ─────────────────────────────────────────
def get_volatility(sl_pips, trade_log):
    recent_sl = [t.get("sl_pips", 0) for t in trade_log[-20:] if t.get("sl_pips", 0) > 0]
    if len(recent_sl) < 3:
        return 1.0
    avg = sum(recent_sl) / len(recent_sl)
    return round(sl_pips / avg, 3) if avg > 0 else 1.0


# ── Rolling stats ─────────────────────────────────────────────
def get_rolling_stats(trade_log):
    labeled = [t for t in trade_log if t.get("result") in ("win", "loss")]
    if not labeled:
        return 0.5, 0
    last10  = labeled[-10:]
    wr      = sum(1 for t in last10 if t["result"] == "win") / len(last10)
    streak  = 0
    for t in reversed(labeled):
        if t["result"] == "loss":
            streak += 1
        else:
            break
    return round(wr, 3), min(streak, 5)


# ── Feature extraction ────────────────────────────────────────
def extract_features(parsed, trade_log):
    zone  = normalise_zone(parsed.get("zone"))
    ze    = ZONE_MAP[zone]
    conf  = min(int(parsed.get("conf", "1")), 4)
    cmd   = 1 if parsed.get("command", "").lower() == "buy" else 0
    sl    = float(parsed.get("sl", "50") or "50")
    tp    = float(parsed.get("tp", "100") or "100")
    rr    = round(tp / sl, 3) if sl > 0 else 2.0
    risk  = float(parsed.get("risk", "1.0") or "1.0")
    now   = datetime.now(timezone.utc)
    hour  = now.hour
    dow   = now.weekday()
    sess  = get_session(hour)
    vol   = get_volatility(sl, trade_log)
    rwr, lstreak = get_rolling_stats(trade_log)
    return [ze, conf, cmd, sl, tp, rr, risk, hour, dow, sess, vol, rwr, lstreak]


# ── Alert parsing ─────────────────────────────────────────────
def parse_alert(alert_str):
    parts = [p.strip() for p in alert_str.split(",")]
    result = {
        "license_id": parts[0] if len(parts) > 0 else LICENSE_ID,
        "command":    parts[1] if len(parts) > 1 else "",
        "symbol":     parts[2] if len(parts) > 2 else "XAUUSD.s",
        "raw":        alert_str,
        "named":      {}
    }
    for part in parts[3:]:
        if "=" in part:
            k, v = part.split("=", 1)
            result["named"][k.strip().lower()] = v.strip()
    result["sl"]   = result["named"].get("sl",   "50")
    result["tp"]   = result["named"].get("tp",   "100")
    result["risk"] = result["named"].get("risk", "1.0")
    result["zone"] = result["named"].get("zone", "UNKNOWN")
    result["conf"] = result["named"].get("conf", "1")
    return result

def adjust_sl(sl_str):
    try:
        return str(round(max(float(sl_str), GOLD_MIN_SL_PIPS), 1))
    except: return sl_str

def rebuild_alert(parsed):
    named = parsed["named"].copy()
    named.pop("zone", None)
    named.pop("conf", None)
    named["sl"] = adjust_sl(parsed["sl"])
    param_str = ",".join(f"{k}={v}" for k, v in named.items())
    return f"{parsed['license_id']},{parsed['command']},{parsed['symbol']},{param_str}"


# ── Training ──────────────────────────────────────────────────
def get_xy(trade_log):
    X, y = [], []
    for t in trade_log:
        if t.get("result") not in ("win", "loss"): continue
        if "features" not in t: continue
        X.append(t["features"])
        y.append(1 if t["result"] == "win" else 0)
    return np.array(X), np.array(y)

def retrain_rf(trade_log, sc):
    X, y = get_xy(trade_log)
    if len(X) < RF_MIN_SAMPLES:
        return None, sc
    if len(set(y.tolist())) < 2:
        log.warning("[RF] Only one class present. Skipping retrain.")
        return None, sc
    wins = int(y.sum()), len(y) - int(y.sum())
    log.info(f"[RF] Retraining on {len(X)} trades - {wins[0]}W {wins[1]}L")
    sc_new = StandardScaler()
    X_sc   = sc_new.fit_transform(X)
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_sc, y)
    acc = round(clf.score(X_sc, y) * 100, 1)
    log.info(f"[RF] Train accuracy: {acc}%")
    return clf, sc_new

def update_sgd(sgd, sc, new_features, new_label):
    """Incremental update after every single trade."""
    X_new = np.array([new_features])
    y_new = np.array([new_label])
    if sgd is None:
        # class_weight="balanced" is NOT supported by partial_fit and raises
        # every time. Left as None; class imbalance is handled by the dynamic
        # threshold in get_threshold() instead.
        sgd = SGDClassifier(
            loss="log_loss", learning_rate="optimal",
            random_state=42)
    try:
        # StandardScaler supports partial_fit, so the scaler learns online too.
        # Without this the scaler stays unfitted until RF's first retrain at
        # RF_MIN_SAMPLES, which blocks SGD from ever warming up.
        sc.partial_fit(X_new)
        X_sc = sc.transform(X_new)
        sgd.partial_fit(X_sc, y_new, classes=np.array([0, 1]))
        log.info(f"[SGD] Online update label={new_label}")
    except Exception as e:
        log.warning(f"[SGD] Update failed: {e}")
    return sgd

def update_sdagd(sdagd, sc, new_features, new_label):
    """
    Shadow-mode online update using the Stochastic Diagonal Approximate
    Greatest Descent classifier (sdagd_classifier.py). Learns from every
    labeled trade exactly like SGD, but its output does not (yet)
    influence predict_win_prob()'s blocking decision -- see that function
    and the SDAGD_LIVE flag below. This lets its track record (phase
    transitions, radius behavior, predicted-vs-actual accuracy) be
    compared against RF/SGD before it's trusted in the live ensemble.
    """
    X_new = np.array([new_features])
    if sdagd is None:
        sdagd = SDAGDClassifier(n_features=len(new_features))
    try:
        # sc.partial_fit already called by update_sgd() in the same /result
        # request (both run against the same scaler instance), so we just
        # transform here rather than fit again.
        X_sc = sc.transform(X_new)
        sdagd.partial_fit(X_sc, np.array([new_label]))
        log.info(f"[SDAGD] Online update label={new_label} "
                 f"phase={sdagd.last_phase} r={sdagd.r:.4f}")
    except Exception as e:
        log.warning(f"[SDAGD] Update failed: {e}")
    return sdagd

def predict_win_prob(rf, sgd, sc, features):
    """
    Ensemble prediction. Falls back gracefully.
    NOTE: SDAGD is intentionally NOT included here yet -- it's shadow-mode
    only (see update_sdagd()). Once its shadow track record is validated,
    fold it in here (e.g. reweight rf/sgd/sdagd, or swap it in for sgd).
    """
    X = np.array([features])
    try: X_sc = sc.transform(X)
    except: return None, "no_scaler"

    rf_prob  = None
    sgd_prob = None

    if rf is not None:
        try: rf_prob = float(rf.predict_proba(X_sc)[0][1])
        except: pass

    if sgd is not None:
        try: sgd_prob = float(sgd.predict_proba(X_sc)[0][1])
        except: pass

    if rf_prob is not None and sgd_prob is not None:
        prob   = round(0.6 * rf_prob + 0.4 * sgd_prob, 4)
        source = "ensemble"
    elif rf_prob is not None:
        prob   = round(rf_prob, 4)
        source = "rf_only"
    elif sgd_prob is not None:
        prob   = round(sgd_prob, 4)
        source = "sgd_only"
    else:
        return None, "no_model"

    return prob, source


def shadow_predict_sdagd(sdagd, sc, features):
    """Compute SDAGD's own probability for logging/comparison purposes
    only. Never used for blocking decisions."""
    if sdagd is None:
        return None
    X = np.array([features])
    try:
        X_sc = sc.transform(X)
        return round(float(sdagd.predict_proba(X_sc)[0][1]), 4)
    except Exception:
        return None


# ── Routes ────────────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def receive_signal():
    t_receipt = time.time()
    raw = request.data.decode("utf-8").strip()
    log.info(f"[IN] {raw[:120]}")

    with lock:
        trade_log           = load_log()
        rf, sgd, sdagd, sc   = load_models()
        parsed        = parse_alert(raw)
        features      = extract_features(parsed, trade_log)
        zone          = normalise_zone(parsed.get("zone"))
        conf          = int(parsed.get("conf", "1"))

        if zone == "UNKNOWN":
            log.warning("[ZONE] Signal carried no recognised zone tag. "
                        "Check the Pine alert template appends zone= and conf=.")

        if parsed["license_id"] != LICENSE_ID:
            log.warning(f"[LICENSE] Payload id {parsed['license_id']} "
                        f"does not match configured {LICENSE_ID}. "
                        f"PineConnector will ignore this order.")

        labeled       = [t for t in trade_log if t.get("result") in ("win","loss")]
        threshold     = get_threshold(trade_log)
        win_prob, src = predict_win_prob(rf, sgd, sc, features)
        sdagd_prob    = shadow_predict_sdagd(sdagd, sc, features)  # shadow-only, logged not used

        blocked = win_prob is not None and win_prob < threshold
        explored = False

        if blocked and len(labeled) < MIN_LABELED_TO_BLOCK:
            blocked = False
            explored = True
            log.info(f"[WARMUP] {zone} prob={win_prob} below {threshold} "
                     f"but only {len(labeled)}/{MIN_LABELED_TO_BLOCK} labeled. Passing.")
        elif blocked and random.random() < EXPLORE_RATE:
            blocked = False
            explored = True
            log.info(f"[EXPLORE] {zone} prob={win_prob} below {threshold}. "
                     f"Passing anyway to keep the loop learning.")

        if sdagd_prob is not None:
            log.info(f"[SDAGD-SHADOW] {zone} prob={sdagd_prob} "
                     f"(live decision used {src} prob={win_prob})")

        entry = {
            "id":        datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
            "timestamp": utc_now_iso(),
            "zone":      zone,
            "confluence": conf,
            "command":   parsed["command"],
            "symbol":    parsed["symbol"],
            "sl_pips":   float(parsed.get("sl","50") or "50"),
            "tp_pips":   float(parsed.get("tp","100") or "100"),
            "risk":      parsed.get("risk","1.0"),
            "features":  features,
            "win_prob":  win_prob,
            "sdagd_shadow_prob": sdagd_prob,
            "threshold": threshold,
            "decision":  "explore" if explored else (src if win_prob is not None else "pass-through"),
            "is_layer":  zone == "LAYER",
            "payload":   rebuild_alert(parsed),
            "result":    "blocked" if blocked else "pending",
            "pnl":       None,
        }

        if blocked:
            log.warning(f"[BLOCK] {zone} prob={win_prob} < thresh={threshold} src={src}")
            trade_log.append(entry)
            save_log(trade_log)
            return jsonify({
                "status":    "blocked",
                "zone":      zone,
                "win_prob":  win_prob,
                "threshold": threshold,
                "source":    src,
            }), 200

        if win_prob is not None:
            log.info(f"[PASS] {zone} prob={win_prob} >= {threshold} src={src}")
        else:
            log.info(f"[PASS] {zone} pass-through ({len(labeled)} labeled trades)")

        trade_log.append(entry)
        save_log(trade_log)

        try:
            resp = requests.post(PINECONNECTOR_URL, data=entry["payload"],
                                 headers={"Content-Type": "text/plain"}, timeout=5)
            relay_ms = round((time.time() - t_receipt) * 1000, 1)
            log.info(f"[OUT] {resp.status_code} | relay={relay_ms}ms")
            return jsonify({
                "status":    "forwarded",
                "id":        entry["id"],
                "zone":      zone,
                "win_prob":  win_prob,
                "threshold": threshold,
                "decision":  entry["decision"],
                "labeled":   len(labeled),
                "relay_ms":  relay_ms,
            }), 200
        except Exception as e:
            log.error(f"[ERR] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/match", methods=["GET"])
def match_fill():
    """
    Called by MTL_ResultReporter. Given a fill time as a UTC epoch, return
    the nearest still-pending signal inside the window.

    GET /match?ts=1751030400&window=180
    -> {"id": "2026...", "zone": "FVG", "delta": 12}
    -> {"id": null, "reason": "no_candidate", "nearest_delta": 7203}
    """
    try:
        ts = int(request.args.get("ts", "0"))
    except ValueError:
        return jsonify({"id": None, "reason": "bad_ts"}), 400

    window = int(request.args.get("window", MATCH_WINDOW_SEC))

    with lock:
        trade_log = load_log()

    best, best_abs = None, None
    nearest_abs = None

    for t in trade_log:
        if t.get("result") != "pending":
            continue
        sig = iso_to_epoch(t.get("timestamp", ""))
        if sig is None:
            continue

        delta = ts - sig                    # positive = fill after signal
        a = abs(delta)
        if nearest_abs is None or a < nearest_abs:
            nearest_abs = a

        # A fill should follow its signal. Allow a small lead for clock skew.
        if delta < -MATCH_LEAD_SEC or delta > window:
            continue

        if best_abs is None or a < best_abs:
            best_abs, best = a, t

    if best is None:
        log.warning(f"[MATCH] No pending signal within {window}s of {ts}. "
                    f"Nearest was {nearest_abs}s away.")
        return jsonify({
            "id": None,
            "reason": "no_candidate",
            "nearest_delta": nearest_abs,
            "window": window,
        }), 200

    log.info(f"[MATCH] {best['id']} zone={best['zone']} delta={ts - iso_to_epoch(best['timestamp'])}s")
    return jsonify({
        "id":    best["id"],
        "zone":  best.get("zone", "UNKNOWN"),
        "delta": ts - iso_to_epoch(best["timestamp"]),
    }), 200


@app.route("/result", methods=["POST"])
def receive_result():
    data     = request.json or {}
    result   = str(data.get("result", "")).lower()
    pnl      = data.get("pnl", 0)
    trade_id = str(data.get("trade_id", "")).strip()

    if result not in ("win", "loss"):
        return jsonify({"status": "error", "message": "result must be win or loss"}), 400
    if not trade_id:
        log.warning("[RESULT] Rejected: empty trade_id.")
        return jsonify({"status": "error", "message": "trade_id required"}), 409

    log.info(f"[RESULT] id={trade_id} result={result} pnl={pnl}")

    with lock:
        trade_log         = load_log()
        rf, sgd, sdagd, sc = load_models()

        target = None
        for entry in trade_log:
            if entry.get("id") == trade_id:
                target = entry
                break

        if target is None:
            log.warning(f"[RESULT] Rejected: id {trade_id} not found.")
            return jsonify({"status": "error", "message": "trade_id not found"}), 409

        if target.get("result") in ("win", "loss"):
            log.warning(f"[RESULT] Rejected: id {trade_id} already labeled "
                        f"{target['result']}.")
            return jsonify({"status": "error", "message": "already labeled"}), 409

        target["result"] = result
        target["pnl"]    = pnl
        save_log(trade_log)

        if "features" in target:
            label = 1 if result == "win" else 0
            sgd   = update_sgd(sgd, sc, target["features"], label)
            sdagd = update_sdagd(sdagd, sc, target["features"], label)

        labeled = [t for t in trade_log if t.get("result") in ("win","loss")]
        if len(labeled) >= RF_MIN_SAMPLES and len(labeled) % RF_RETRAIN_EVERY == 0:
            rf, sc = retrain_rf(trade_log, sc)

        save_models(rf, sgd, sdagd, sc)

    return jsonify({
        "status":       "ok",
        "id":           trade_id,
        "labeled":      len(labeled),
        "rf_active":    rf    is not None,
        "sgd_active":   sgd   is not None,
        "sdagd_active": sdagd is not None,
        "threshold":    get_threshold(trade_log),
    }), 200


@app.route("/model", methods=["GET"])
def get_model():
    with lock:
        trade_log          = load_log()
        rf, sgd, sdagd, sc = load_models()

    labeled = [t for t in trade_log if t.get("result") in ("win","loss")]
    wins    = sum(1 for t in labeled if t["result"] == "win")
    losses  = len(labeled) - wins

    info = {
        "rf_active":    rf    is not None,
        "sgd_active":   sgd   is not None,
        "sdagd_active": sdagd is not None,
        "sdagd_phase":  sdagd.last_phase if sdagd is not None else None,
        "sdagd_radius": round(sdagd.r, 4) if sdagd is not None else None,
        "sdagd_updates": sdagd.n_updates if sdagd is not None else 0,
        "labeled":      len(labeled),
        "wins":         wins,
        "losses":       losses,
        "win_rate_pct": round(wins/len(labeled)*100,1) if labeled else 0,
        "threshold":    get_threshold(trade_log),
        "rf_min":       RF_MIN_SAMPLES,
        "sgd_min":      SGD_MIN_SAMPLES,
        "features":     FEATURE_NAMES,
    }

    if rf is not None:
        X, y = get_xy(trade_log)
        if len(X) > 0:
            try:
                X_sc = sc.transform(X)
                info["rf_train_accuracy"] = round(rf.score(X_sc, y)*100, 1)
                imp = {FEATURE_NAMES[i]: round(float(v),4)
                       for i, v in enumerate(rf.feature_importances_)}
                info["feature_importance"] = dict(sorted(imp.items(), key=lambda x: -x[1]))
            except: pass

    if sdagd is not None:
        X, y = get_xy(trade_log)
        if len(X) > 0:
            try:
                X_sc = sc.transform(X)
                preds = sdagd.predict(X_sc)
                info["sdagd_shadow_accuracy"] = round(float((preds == y).mean()) * 100, 1)
            except Exception:
                pass

    return jsonify(info), 200


@app.route("/stats", methods=["GET"])
def get_stats():
    with lock: trade_log = load_log()
    stats = {}
    for zone in ZONE_CLASSES:
        trades  = [t for t in trade_log if t.get("zone") == zone]
        labeled = [t for t in trades if t.get("result") in ("win","loss")]
        wins    = sum(1 for t in labeled if t["result"] == "win")
        blocked = sum(1 for t in trades if t.get("result") == "blocked")
        pending = sum(1 for t in trades if t.get("result") == "pending")
        pnl     = sum(t.get("pnl",0) or 0 for t in labeled)
        stats[zone] = {
            "signals": len(trades), "labeled": len(labeled),
            "wins": wins, "losses": len(labeled)-wins,
            "win_rate_pct": round(wins/len(labeled)*100,1) if labeled else 0,
            "pnl": round(pnl,2), "blocked": blocked, "pending": pending,
        }
    return jsonify(stats), 200


@app.route("/log", methods=["GET"])
def get_log():
    with lock: return jsonify(load_log()), 200


@app.route("/params", methods=["GET"])
def get_params():
    with lock: trade_log = load_log()
    return jsonify({
        "rf_min_samples":    RF_MIN_SAMPLES,
        "sgd_min_samples":   SGD_MIN_SAMPLES,
        "rf_retrain_every":  RF_RETRAIN_EVERY,
        "current_threshold": get_threshold(trade_log),
        "gold_min_sl_pips":  GOLD_MIN_SL_PIPS,
        "match_window_sec":  MATCH_WINDOW_SEC,
        "feature_names":     FEATURE_NAMES,
        "zone_classes":      ZONE_CLASSES,
        "license_id":        LICENSE_ID,
        "base_dir":          BASE_DIR,
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    with lock:
        save_log([])
        for f in [RF_FILE, SGD_FILE, SDAGD_FILE, SCALER_FILE]:
            if os.path.exists(f): os.remove(f)
    log.info("[RESET] Model and log cleared")
    return jsonify({"status": "reset"}), 200


@app.route("/health", methods=["GET"])
def health():
    with lock: trade_log = load_log()
    labeled = [t for t in trade_log if t.get("result") in ("win","loss")]
    pending = [t for t in trade_log if t.get("result") == "pending"]
    rf, sgd, sdagd, _ = load_models()
    return jsonify({
        "status":       "ok",
        "timestamp":    utc_now_iso(),
        "rf_active":    rf    is not None,
        "sgd_active":   sgd   is not None,
        "sdagd_active": sdagd is not None,
        "labeled":      len(labeled),
        "pending":      len(pending),
        "threshold":    get_threshold(trade_log),
    }), 200


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MTL Zone ML Relay v5 (RF + Online SGD + SDAGD shadow) - port 5000")
    log.info(f"Data directory: {BASE_DIR}")
    log.info(f"License: {LICENSE_ID}")
    log.info(f"RF activates after {RF_MIN_SAMPLES} trades, retrains every {RF_RETRAIN_EVERY}")
    log.info("SDAGD runs in SHADOW mode only -- see update_sdagd()/predict_win_prob()")
    log.info("GET  /health  -> status")
    log.info("GET  /model   -> accuracy + feature importance + SDAGD shadow stats")
    log.info("GET  /stats   -> per-zone breakdown")
    log.info("GET  /match   -> fill-to-signal matching for the reporter EA")
    log.info("POST /reset   -> wipe everything")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)