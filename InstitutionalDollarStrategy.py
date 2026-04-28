"""
InstitutionalDollarStrategy.py  (Phase 8 — ZMQ Subscriber Shell)

Architecture:
    market_data_daemon.py  ──ZMQ PUB──►  InstitutionalDollarStrategy
                                              └── _ZmqListener thread
                                                      │
                                                      ▼  (on each Dollar Bar)
                                                  Features → LightGBM → HMM → Kelly
                                                      │
                                                      ▼
                                              _latest_signal (thread-safe dict)
                                                      │
                                              populate_indicators()
                                                      │
                                              populate_entry/exit_trend()
                                                      │
                                              Freqtrade order routing
                                          custom_stake_amount / custom_entry_price

This strategy is a THIN SHELL.  All data ingestion and intelligence computation
happen in the daemon + the background listener thread, NOT in the Freqtrade
polling loop.  Freqtrade only manages the lifecycle of individual orders.

WARNING: Not designed for native Freqtrade backtesting.
         Run exclusively with --dry-run or --live.
"""

import queue
import sys
import json
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# ── Path bootstrap: must run BEFORE any third-party imports ──────────────────
# Freqtrade uses importlib.util which does NOT inherit sys.path from the caller.
# We resolve RL_Trading/ relative to THIS file's real location so that copies
# placed in strategy subfolders (e.g. strategies/institutional/) still work.
_THIS_FILE  = Path(__file__).resolve()
_STRAT_DIR  = _THIS_FILE.parent
# Support both: direct in strategies/ OR in strategies/institutional/
_RL_DIR = _STRAT_DIR / "RL_Trading"
if not _RL_DIR.exists():
    _RL_DIR = _STRAT_DIR.parent / "RL_Trading"

for _p in [str(_RL_DIR), str(_RL_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter

from utils.signal_features import compute_ohlcv_features
from utils.risk_directors  import MahalanobisTurbulence, HMMRegimeModel
from utils.position_sizer  import FractionalKellySizer
from utils.filters         import SymmetricCUSUMFilter

logger = logging.getLogger(__name__)

# ── ZMQ daemon topics (must match market_data_daemon.py) ─────────────────────
TOPIC_DOLLAR_BAR  = b"DOLLAR_BAR"
TOPIC_BOOK_TICKER = b"BOOK_TICKER"
ZMQ_DAEMON_ADDR   = "tcp://127.0.0.1:5555"
ZMQ_DIAG_ADDR     = "tcp://127.0.0.1:5556"   # Dashboard diagnostics PUB

# Minimum Dollar Bars in the rolling buffer before running inference
# Lowered to 20: enough for RSI/vol warmup while keeping cold-start fast.
MIN_BARS_FOR_INFERENCE = 20


class _ZmqListener(threading.Thread):
    """
    Background daemon thread that subscribes to the ZMQ PUB socket and drives
    the full intelligence pipeline (features → LightGBM → HMM → sizing) each
    time a completed Dollar Bar arrives.  Results are written atomically into
    a shared signal dict protected by a threading.Lock.
    """

    def __init__(self, alpha_model, meta_model, turbulence_engine, hmm_model, hmm_model_htf, sizer,
                 signal_store: dict, lock: threading.Lock,
                 zmq_addr: str = ZMQ_DAEMON_ADDR):
        super().__init__(daemon=True, name="ZmqDollarBarListener")
        self._alpha         = alpha_model
        self._meta          = meta_model
        self._turb          = turbulence_engine
        self._hmm           = hmm_model
        self._hmm_htf       = hmm_model_htf   # 1-hour timeframe HMM
        self._sizer         = sizer
        self._signal        = signal_store
        self._lock          = lock
        self._zmq_addr      = zmq_addr
        self._bar_buffer: list[dict] = []
        self._running       = True

        # Clock Decoupling State
        self._cusum = SymmetricCUSUMFilter()
        self._current_target_pos = 0.0

        # HMM Robust Inference State (LTF — dollar bar resolution)
        self._hmm_fitted = False
        self._bars_since_fit = 0

        # HTF HMM state (1-hour aggregated bars)
        self._htf_fitted = False
        self._htf_bars_since_fit = 0

        # D-02 FIX: Temporal hysteresis for regime label.
        # Asymmetric: faster into bull (miss fewer rallies), slower into bear (avoid whipsaw).
        # Long-only: false negatives (blocked bull) cost 100% of the rally;
        # false positives (wrong bull) only cost fees if it reverses quickly.
        self._HYSTERESIS_TO_BULL = 3   # bear→bull: 3 bars confirmation (restored; 2 was over-firing)
        self._HYSTERESIS_TO_BEAR = 6   # bull→bear: require more confirmation
        self._pending_regime: Optional[str] = None
        self._pending_regime_count: int = 0
        self._committed_regime: str = "unknown"

        # BYPASS HOLD: After a bypass entry, sustain the 10% floor for up to
        # _HYSTERESIS_TO_BULL bars so the position isn't killed before the HMM
        # commits.  Cleared early if pending regime pivots away from bull or
        # alpha falls below 0.45 (model reversed).
        self._bypass_hold_bars: int = 0
        
        # BUG #23 FIX: Coalescing state
        self._pipeline_inflight = False
        self._inflight_since = 0.0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._last_t_processed = 0.0

        # Dashboard diagnostics: fire-and-forget queue (thread-safe)
        self._diag_queue: queue.SimpleQueue = queue.SimpleQueue()

        # Institutional Stats Monitoring (Shadow-Live)
        self._stats = {
            "n_inferences": 0,
            "alpha_probs": [],
            "meta_probs": [],
            "spreads": [],
            "costs": [],
            "veto_count": 0,
            "entry_count": 0,
            "reasons": {}
        }


    def set_virtual_position(self, pos: float):
        """BUG #19 FIX: Thread-safe update of current position."""
        with self._lock:
            self._current_target_pos = float(pos)

    def stop(self):
        self._running = False
        if self._executor:
            # BUG-Shutdown FIX: non-waiting shutdown
            self._executor.shutdown(wait=False)

    def _log_summary_stats(self):
        """Institutional Summary Log (Shadow-Live Audit)"""
        s = self._stats
        if s["n_inferences"] == 0: return
        
        alpha_arr = np.array(s["alpha_probs"])
        meta_arr  = np.array(s["meta_probs"])
        
        logger.info(
            "\n" + "="*60 + "\n"
            f"[SHADOW-LIVE SUMMARY] Bars: {s['n_inferences']}\n"
            f"Alpha Prob: Mean={np.mean(alpha_arr):.3f} | P10={np.percentile(alpha_arr, 10):.3f} | P50={np.median(alpha_arr):.3f} | P90={np.percentile(alpha_arr, 90):.3f}\n"
            f"Meta Prob:  Mean={np.mean(meta_arr):.3f}  | P10={np.percentile(meta_arr, 10):.3f}  | P50={np.median(meta_arr):.3f}  | P90={np.percentile(meta_arr, 90):.3f}\n"
            f"Veto Rate:  {(s['veto_count']/s['n_inferences'])*100:.1f}% | Entry Rate: {(s['entry_count']/s['n_inferences'])*100:.1f}%\n"
            f"Spreads:    Mean={np.mean(s['spreads']):.1f} bps | Costs: Mean={np.mean(s['costs']):.1f} bps\n"
            f"Veto Reasons: {s['reasons']}\n"
            + "="*60
        )
        # Clear rolling window to avoid memory leak over days, but keep counts
        s["alpha_probs"] = []
        s["meta_probs"] = []
        s["spreads"] = []
        s["costs"] = []

    def _run_pipeline(self):
        """Converts bar buffer → features → alpha → regime → sizing."""
        try:
            with self._lock:
                buffer_copy = list(self._bar_buffer[-2000:])
                
            if len(buffer_copy) < MIN_BARS_FOR_INFERENCE:
                return

            df = pd.DataFrame(buffer_copy).reset_index(drop=True)

            # Feature engineering
            from utils.signal_features import build_feature_matrix, SIGNAL_FEAT_COLS_V2
            try:
                X = build_feature_matrix(df.copy(), eth_df=None, funding_series=None)
            except Exception as e:
                logger.warning("Feature engineering failed: %s", e, exc_info=True)
                return

            if len(X) < 20:
                return

            # Alpha Orcale inference (Fast + Slow Stacking)
            try:
                # 1. Fast Alpha (Dollar Bars)
                model_features = self._alpha.feature_name()
                X_fast = X[model_features].fillna(0.0)
                fast_preds = self._alpha.predict(X_fast)
                df["oof_pred_fast"] = pd.Series(fast_preds).fillna(0.5).clip(0.0, 1.0)

                # 2. Slow Alpha (1h Ensemble Logic)
                alpha_slow = 0.5
                if self._alpha_slow and isinstance(self._alpha_slow, dict):
                    slow_models = self._alpha_slow.get("models", {})
                    weights = self._alpha_slow.get("weights", {})
                    logits = 0.0
                    total_w = 0.0
                    last_idx = len(X) - 1
                    for h, model in slow_models.items():
                        slow_feats = getattr(model, "feature_name", lambda: [])()
                        # Align features
                        X_s = X[slow_feats].fillna(0.0) if slow_feats else X
                        # LGBMClassifier needs predict_proba
                        p = model.predict_proba(X_s)[last_idx, 1]
                        p = np.clip(p, 1e-6, 1-1e-6)
                        w = weights.get(h, 1.0)
                        logits += w * np.log(p / (1 - p))
                        total_w += w
                    
                    if total_w > 0:
                        alpha_slow = 1.0 / (1.0 + np.exp(-logits / total_w))
                
                df["oof_pred_slow"] = alpha_slow
                
                # 3. Stacked Prediction: Requirement of cross-timeframe agreement
                last_fast = float(df["oof_pred_fast"].iloc[-1])
                if abs(alpha_slow - 0.5) < 0.02: 
                    # If slow signal is neutral, we dampen the fast signal significantly
                    df["oof_pred"] = 0.5 + (last_fast - 0.5) * 0.3
                else:
                    is_bull = (alpha_slow > 0.51) and (last_fast > 0.51)
                    is_bear = (alpha_slow < 0.49) and (last_fast < 0.49)
                    if not (is_bull or is_bear):
                        df["oof_pred"] = 0.5 # kill signal
                    else:
                        df["oof_pred"] = last_fast

            except Exception as e:
                logger.warning("Combined Alpha inference failed: %s", e, exc_info=True)
                return

            # Meta-model "Gatekeeper" Features (Context-Aware)
            try:
                # 1. Feature Prep
                alpha_probs = df["oof_pred"]
                df["alpha_prob_smooth"] = alpha_probs.ewm(span=10).mean()
                df["alpha_prob_zscore"] = (alpha_probs - alpha_probs.rolling(200).mean()) / alpha_probs.rolling(200).std()
                df["alpha_prob_percentile"] = alpha_probs.rolling(1000).rank(pct=True)
                df["alpha_signal_persistence"] = (alpha_probs > 0.55).astype(int).rolling(20).sum()
                
                # We'll compute turbulence percentile later after full turb computation
            except Exception as e:
                logger.warning("Meta-feature prep failed: %s", e)

            # Turbulence (rolling adaptive threshold)
            risk_vec       = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
            available_risk = [c for c in risk_vec if c in X.columns]
            if len(available_risk) < 2:
                return

            # Attach features back to df for turbulence / HMM
            for col in available_risk + ["log_return_feature", "volatility_24_feature"]:
                if col in X.columns:
                    df[col] = X[col].values

            turb_series = self._turb.compute(df, available_risk)
            # D-05 FIX: NaN turbulence (first 1000-bar warmup) should NOT be filled with 0.
            # 0 turbulence = no penalty = full Kelly exposure during the riskiest cold-start period.
            # Fill with the 95th percentile of observed values (conservative) or a fixed high value.
            if turb_series is not None:
                turb_p95 = turb_series.quantile(0.95) if turb_series.notna().sum() > 10 else 5.0
                turb_p95 = float(turb_p95) if not np.isnan(turb_p95) else 5.0
                df["turbulence_score"] = turb_series.fillna(turb_p95)
            else:
                df["turbulence_score"] = 5.0  # conservative default

            adaptive_thr = self._turb.rolling_threshold(df["turbulence_score"])

            # HMM Regime (Priority 3: Robust Online Inference)
            hmm_feats = ["log_return_feature", "volatility_24_feature"]
            # D-09 FIX: Expand HMM to use same 4D space as Mahalanobis turbulence.
            # aggressor_ratio and intraday_range_feature add microstructure signal
            # that distinguishes clean bull_calm from bull_calm with selling pressure.
            extended_feats = [
                "log_return_feature", "volatility_24_feature",
                "aggressor_ratio", "intraday_range_feature",
            ]
            hmm_feats = [f for f in extended_feats if f in df.columns] or hmm_feats
            available_hmm = [c for c in hmm_feats if c in df.columns]
            
            try:
                if not self._hmm_fitted or self._bars_since_fit >= 500:
                    logger.info("[HMM] Re-fitting model (interval=500 bars, window=%d)...", len(df))
                    self._hmm.fit(df, available_hmm)
                    self._hmm_fitted = True
                    self._bars_since_fit = 0
                
                # Predict current state using the last 50 observations
                # Semantic labeling (Log Return is features[0])
                assert self._hmm.features[0] == "log_return_feature", "HMM first feature must be log_return_feature for semantic labeling"
                current_regime_raw = self._hmm.predict_current(df.tail(50))

                # D-02 FIX: Temporal hysteresis — only commit regime change after
                # N consecutive bars with the same new label (asymmetric: faster into bull).
                _bull_regimes = {"bull_calm", "bull_neutral", "high_vol_rebound"}
                _hysteresis_needed = (
                    self._HYSTERESIS_TO_BULL if current_regime_raw in _bull_regimes
                    else self._HYSTERESIS_TO_BEAR
                )
                if current_regime_raw == self._committed_regime:
                    self._pending_regime = None
                    self._pending_regime_count = 0
                elif current_regime_raw == self._pending_regime:
                    self._pending_regime_count += 1
                    if self._pending_regime_count >= _hysteresis_needed:
                        logger.info(
                            "[REGIME] Committed %s → %s after %d bars",
                            self._committed_regime, current_regime_raw,
                            self._pending_regime_count
                        )
                        self._committed_regime = current_regime_raw
                        self._pending_regime = None
                        self._pending_regime_count = 0
                else:
                    self._pending_regime = current_regime_raw
                    self._pending_regime_count = 1

                current_regime = self._committed_regime
                df["hmm_semantic_regime"] = current_regime
                self._bars_since_fit += 1
            except Exception as e:
                # BUG #22 FIX: exc_info=True
                logger.warning("HMM online inference failed: %s", e, exc_info=True)
                df["hmm_semantic_regime"] = "unknown"

            last = df.iloc[-1]

            # 5. Kelly Portfolio Sizing (The "Raw" Signal)
            # T2.7 Daily Volatility for Barrier Scaling
            # Labels are pt=1.5 * daily_vol; Kelly must match this scale to overcome costs.
            log_ret = df["log_return_feature"]
            daily_vol = log_ret.rolling(self._WINDOW_DAILY, min_periods=50).std() * np.sqrt(self._WINDOW_DAILY)
            last_daily_vol = float(daily_vol.iloc[-1]) if not np.isnan(daily_vol.iloc[-1]) else 0.02
            barrier_height = max(0.003, 1.5 * last_daily_vol)

            raw_target_pos = self._sizer.size_portfolio(
                probabilities=df["oof_pred"],
                regimes=df["hmm_semantic_regime"],
                turbulence=df["turbulence_score"],
                adaptive_threshold=adaptive_thr,
                risk_scales=pd.Series(barrier_height, index=df.index),
                dof=len(available_risk),
            ).iloc[-1]

            # T2.4: ATR14 — trailing stop anchor (EWM for responsiveness)
            atr14 = None
            try:
                h_arr  = df["high"].values.astype(float)
                l_arr  = df["low"].values.astype(float)
                c_arr  = df["close"].values.astype(float)
                pc_arr = np.concatenate([[c_arr[0]], c_arr[:-1]])
                tr_arr = np.maximum(h_arr - l_arr,
                         np.maximum(np.abs(h_arr - pc_arr), np.abs(l_arr - pc_arr)))
                atr14  = float(pd.Series(tr_arr).ewm(span=14, adjust=False).mean().iloc[-1])
                if not np.isfinite(atr14):
                    atr14 = None
            except Exception:
                pass

            # T2.5-Phase 5: Stacking and Regime Hardening
            last_oof = float(last["oof_pred"])
            
            # Kill-Switch 1: Regime Based
            # Total blackout in high-risk regimes for Longs
            hard_blackout = {"unknown", "panic_selloff", "bear_neutral"}
            if str(last["hmm_semantic_regime"]) in hard_blackout:
                raw_target_pos = 0.0
                logger.info("[KILL-SWITCH] Regime %s -> Position zeroed", last["hmm_semantic_regime"])
            
            # Kill-Switch 2: Volume Cluster Confirmation
            # Z-score of 24 bars. 1.0 means volume is 1 std above mean.
            vol_z = float(X["volume_zscore_24_feature"].iloc[-1]) if "volume_zscore_24_feature" in X.columns else 0.0
            if last_oof > 0.52 and vol_z < 0.5: # signal but no volume pop
                raw_target_pos *= 0.5 # reduce half
                logger.info("[KILL-SWITCH] Weak Volume (z=%.2f) -> Position halved", vol_z)

            # Kill-Switch 3: Long-Only Filter
            if last_oof < 0.50:
                raw_target_pos = 0.0
                logger.info("[KILL-SWITCH] Long-Only Filter active (oof_pred < 0.50) -> Position zeroed")

            confidence_scale = min(1.0, (2.0 * abs(last_oof - 0.5)) / 0.10)
            raw_target_pos = float(raw_target_pos) * confidence_scale

            # Kill-Switch 4: Meta-Model Gatekeeper (Tradeability)
            meta_prob = 0.0
            if self._meta:
                try:
                    # Capture latest book data for spread calculation
                    with self._lock:
                        l_sig = dict(self._signal)
                    
                    # Calculate final meta-features
                    df["turbulence_percentile"] = df["turbulence_score"].rolling(2000).rank(pct=True)
                    
                    # Costs
                    spread_bps = 0.0
                    mid = l_sig.get("mid", 0.0)
                    if l_sig.get("best_ask") and l_sig.get("best_bid") and mid > 0:
                         spread_bps = (l_sig["best_ask"] - l_sig["best_bid"]) / mid * 10000
                    
                    # If live book ticker not in sig yet, fallback to 2 bps spread estimate
                    spread_bps = spread_bps if spread_bps > 0 else 2.0
                    expected_cost_bps = 5 + spread_bps + 2 # 5 fee + spread + 2 slippage
                    
                    # Get canonical HMM state ID (0, 1, 2)
                    hmm_state_id = self._hmm.predict_current_state(df.tail(50))
                    
                    # Map names to match meta-model training script
                    X_meta = pd.DataFrame([{
                        "alpha_prob":              float(last_oof), # Raw probability
                        "alpha_prob_smooth":       float(df["alpha_prob_smooth"].iloc[-1]),
                        "alpha_prob_zscore":       float(df["alpha_prob_zscore"].iloc[-1]),
                        "alpha_prob_percentile":   float(df["alpha_prob_percentile"].iloc[-1]),
                        "alpha_signal_persistence": float(df["alpha_signal_persistence"].iloc[-1]),
                        "turbulence_score":        float(df["turbulence_score"].iloc[-1]),
                        "turbulence_percentile":   float(df["turbulence_percentile"].iloc[-1]),
                        "hmm_state":               int(hmm_state_id),
                        "volatility_24_feature":   float(df["volatility_24_feature"].iloc[-1]),
                        "aggressor_ratio":         float(df["aggressor_ratio"].iloc[-1]) if "aggressor_ratio" in df.columns else 0.0,
                        "l2_imbalance_feature":    float(X["l2_imbalance_feature"].iloc[-1]) if "l2_imbalance_feature" in X.columns else 0.0,
                        "spread_bps":              float(spread_bps),
                        "expected_cost_bps":       float(expected_cost_bps)
                    }])
                    
                    meta_prob = self._meta.predict(X_meta)[0]
                    
                    # LOGGING EXHAUSTIVO (Per Inerence)
                    veto_reason = "none"
                    allow_trade = True
                    
                    if meta_prob < 0.60:
                        raw_target_pos = 0.0
                        allow_trade = False
                        veto_reason = "low_meta_conviction"
                        if expected_cost_bps > 25: veto_reason = "prohibitive_costs"
                        elif spread_bps > 15: veto_reason = "toxic_spread"
                        elif float(last_oof) < 0.55: veto_reason = "alpha_weakness"
                    
                    # Update Stats
                    self._stats["n_inferences"] += 1
                    self._stats["alpha_probs"].append(float(last_oof))
                    self._stats["meta_probs"].append(float(meta_prob))
                    self._stats["spreads"].append(float(spread_bps))
                    self._stats["costs"].append(float(expected_cost_bps))
                    if not allow_trade:
                        self._stats["veto_count"] += 1
                        self._stats["reasons"][veto_reason] = self._stats["reasons"].get(veto_reason, 0) + 1
                    
                    # Log Decisivo
                    logger.info(
                        "[SHADOW-LIVE] Inf #%d | Alpha=%.3f (Thr=0.55) | Meta=%.3f (Thr=0.60) | "
                        "Allow=%s | VetoReason=%s | Regime=%s | Turb=%.2f | "
                        "Spread=%.1f bps | Cost=%.1f bps | Pos: %.2f -> %.2f",
                        self._stats["n_inferences"], float(last_oof), float(meta_prob),
                        allow_trade, veto_reason, current_regime, float(last["turbulence_score"]),
                        spread_bps, expected_cost_bps, self._current_target_pos, raw_target_pos
                    )
                    
                    # Summary Every 100 bars
                    if self._stats["n_inferences"] % 100 == 0:
                        self._log_summary_stats()

                except Exception as e:
                    logger.warning("Meta-model inference failed: %s", e)

            # T2.5: Multi-TF regime filter — 1h aggregated HMM provides market context.
            # Acts as a soft multiplier on the LTF Kelly, not a hard gate.
            htf_regime = "unknown"
            htf_mult   = 0.5   # conservative default when HTF not yet fitted
            try:
                if "t_close" in df.columns:
                    df_htf = (
                        df.assign(
                            _hour=pd.to_datetime(df["t_close"], unit="ms", utc=True).dt.floor("1h")
                        )
                        .groupby("_hour", as_index=False)
                        .agg(
                            open              =("open",                   "first"),
                            high              =("high",                   "max"),
                            low               =("low",                    "min"),
                            close             =("close",                  "last"),
                            volume            =("volume",                 "sum"),
                            aggressor_ratio   =("aggressor_ratio",        "mean"),
                            log_return_feature=("log_return_feature",     "sum"),   # sum = total log return
                            volatility_24_feature=("volatility_24_feature","mean"),
                            intraday_range_feature=("intraday_range_feature","mean"),
                        )
                        .reset_index(drop=True)
                    )

                    if len(df_htf) >= 20:
                        htf_feats = [c for c in [
                            "log_return_feature", "volatility_24_feature",
                            "aggressor_ratio", "intraday_range_feature",
                        ] if c in df_htf.columns]

                        if not self._htf_fitted or self._htf_bars_since_fit >= 50:
                            logger.info("[HTF-HMM] Fitting on %d 1h bars...", len(df_htf))
                            self._hmm_htf.fit(df_htf, htf_feats)
                            self._htf_fitted = True
                            self._htf_bars_since_fit = 0

                        assert self._hmm_htf.features[0] == "log_return_feature"
                        htf_regime = self._hmm_htf.predict_current(df_htf.tail(6))
                        self._htf_bars_since_fit += 1

            except Exception as e:
                logger.warning("HTF regime failed: %s", e)

            # V-BOTTOM FIX: HTF as mult+floor instead of pure multiplication.
            # Floor allows minimum passthrough when LTF shows recovery signal,
            # preventing the multiplicative cascade from killing V-bottom entries.
            _HTF_CONFIG = {
                #                   (mult,  floor)  floor = min passthrough if LTF is recovering
                "bull_calm":        (1.00, 0.00),
                "bull_neutral":     (0.90, 0.00),  # increased from 0.80
                "high_vol_rebound": (0.75, 0.05),  # increased from 0.60
                "bear_neutral":     (0.50, 0.00),  # increased from 0.30
                "bear_calm":        (0.30, 0.05),  # increased from 0.20
                "panic_selloff":    (0.00, 0.00),
                "unknown":          (0.50, 0.00),
            }
            htf_mult, htf_floor = _HTF_CONFIG.get(htf_regime, (0.50, 0.00))
            _ltf_recovery = current_regime in ("high_vol_rebound", "bull_calm", "bull_neutral")
            if htf_mult < 1.0:
                old_pos = raw_target_pos
                multiplicative = raw_target_pos * htf_mult
                if _ltf_recovery and htf_floor > 0:
                    raw_target_pos = max(multiplicative, min(raw_target_pos, htf_floor))
                else:
                    raw_target_pos = multiplicative
                logger.debug(
                    "[HTF] regime=%s mult=%.2f floor=%.2f ltf_recovery=%s → pos %.4f→%.4f",
                    htf_regime, htf_mult, htf_floor, _ltf_recovery, old_pos, raw_target_pos,
                )

            # 6. Event-Driven Hysteresis & Clock Decoupling
            # BUG CUSUM Freeze FIX: Use timestamps instead of local indices to avoid 
            # freezing when the buffer saturates and shifts.
            last_ts = float(df.iloc[-1].get("t_close", 0))
            if self._last_t_processed == 0:
                # Cold start: assume the previous bar was processed
                self._last_t_processed = float(df.iloc[-2].get("t_close", 0)) if len(df) >= 2 else 0
                
            is_event = False
            for i in range(1, len(df)):
                t_i = float(df.iloc[i].get("t_close", 0))
                if t_i <= self._last_t_processed:
                    continue
                    
                prev_p = float(df.iloc[i-1]["close"])
                curr_p = float(df.iloc[i]["close"])
                log_ret = np.log(curr_p / prev_p)
                
                # Priority 4: Dynamic threshold h_t = 3.5 * sigma_recent
                # V-BOTTOM FIX: Use MAD-based robust sigma to resist post-crash
                # outlier contamination (crash spike inflates std, blocking rebound detection).
                sigma_series = df["log_return_feature"].iloc[:i+1].tail(100)
                sigma = sigma_series.std()
                sigma_robust = sigma_series.abs().median() * 1.4826  # MAD→σ equivalent
                if not np.isnan(sigma_robust) and sigma_robust > 0:
                    sigma_eff = min(sigma, sigma_robust * 2.0)
                else:
                    sigma_eff = sigma
                cusum_h = 3.5 * sigma_eff if (not np.isnan(sigma_eff) and sigma_eff > 0) else 0.005
                
                if self._cusum.check(log_ret, cusum_h):
                    is_event = True
            
            self._last_t_processed = last_ts

            # Pre-compute threshold value (reused in bypass + diagnostics)
            _thr_val = None
            if adaptive_thr is not None and adaptive_thr.notna().any():
                try:
                    _v = float(adaptive_thr.iloc[-1])
                    if np.isfinite(_v):
                        _thr_val = _v
                except Exception:
                    pass

            # BYPASS HOLD: Sustain the 10% floor on bars following a bypass entry,
            # bridging the HMM lag until the regime commits or the thesis fails.
            # Cancelled early if pending regime pivots away from bullish targets or
            # alpha reverses below 0.45 (model conviction reversed).
            _bypass_applied = False
            _turb_last = float(last["turbulence_score"])
            _turb_ok = (_thr_val is None) or (_turb_last < _thr_val)
            _bar_imbalance = float(last.get("book_imbalance", 0.0))
            _imbalance_ok  = _bar_imbalance >= -0.20

            if self._bypass_hold_bars > 0:
                _bull_targets = {"bull_calm", "bull_neutral", "high_vol_rebound"}
                _hold_regime_ok = (
                    self._pending_regime in _bull_targets
                    or self._committed_regime in _bull_targets
                )
                _hold_alpha_ok = float(last["oof_pred"]) >= 0.45
                if _hold_regime_ok and _hold_alpha_ok:
                    if raw_target_pos < 0.10:
                        raw_target_pos = 0.10
                    self._bypass_hold_bars -= 1
                    logger.debug(
                        "[BYPASS HOLD] Flooring at 10%% — %d bars remaining (regime=%s alpha=%.3f)",
                        self._bypass_hold_bars, self._committed_regime, float(last["oof_pred"]),
                    )
                else:
                    logger.info(
                        "[BYPASS HOLD] Cancelled — pending=%s committed=%s alpha=%.3f",
                        self._pending_regime, self._committed_regime, float(last["oof_pred"]),
                    )
                    self._bypass_hold_bars = 0

            # ALPHA BYPASS: When HMM is pending a bullish regime (not yet committed),
            # allow a 10% floor position if four independent signals agree:
            #   1. CUSUM fired (structural break)
            #   2. Alpha >= 0.65 (model conviction)
            #   3. Turbulence < adaptive threshold (regime is calm enough)
            #   4. L1 book imbalance >= -0.20 (not heavily sell-side dominated)
            # This bridges the HMM lag on V-bottom entries in long-only mode.
            if (is_event
                    and float(last["oof_pred"]) >= 0.65
                    and _turb_ok
                    and _imbalance_ok
                    and self._pending_regime in ("bull_calm", "bull_neutral", "high_vol_rebound")
                    and raw_target_pos < 0.10):
                raw_target_pos = 0.10
                _bypass_applied = True
                self._bypass_hold_bars = self._HYSTERESIS_TO_BULL  # sustain for N bars
                logger.info(
                    "[ALPHA BYPASS] Pending %s + alpha=%.3f + turb=%.3f < thr=%.3f + imb=%+.3f → floor 10%% (hold=%d bars)",
                    self._pending_regime, float(last["oof_pred"]),
                    _turb_last, _thr_val if _thr_val is not None else float("inf"),
                    _bar_imbalance, self._bypass_hold_bars,
                )

            # Decide whether to commit the new target position
            # Rules: 
            # 1. Must be a CUSUM event (Significant information arrival)
            # 2. Change must be > 1% absolute (Hysteresis gate to avoid jitter)
            # Priority 4: Asymmetric Hysteresis & Execution Discipline
            # Rules (User Spec):
            # 1. Entry: ONLY if target >= 10%
            # 2. Rebalance: ONLY if change >= 10%
            # 3. Exit: Permitted if reduction >= 5% OR target is zero
            # 4. Mandatory: Must be a CUSUM event
            
            pos_diff = abs(raw_target_pos - self._current_target_pos)
            should_update = False
            
            if is_event:
                # Case A: Entry (currently flat)
                if self._current_target_pos == 0:
                    if raw_target_pos >= 0.10:  # raised from 0.05
                        should_update = True
                
                # Case B: Exit / Reduction (reducing exposure)
                elif raw_target_pos < self._current_target_pos:
                    if (pos_diff >= 0.05) or (raw_target_pos == 0):
                        should_update = True
                
                # Case C: Increase / Rebalance
                elif raw_target_pos > self._current_target_pos:
                    if pos_diff >= 0.10:
                        should_update = True
            
            if should_update:
                logger.info(
                    "[EXECUTION EVENT] Pos Update: %.4f -> %.4f (diff=%.4f)",
                    self._current_target_pos, raw_target_pos, pos_diff
                )
                self.set_virtual_position(raw_target_pos)

            # 7. Update Shared Signal Store (Atomic)
            with self._lock:
                self._signal.update({
                    "target_pos":   float(self._current_target_pos),
                    "regime":       str(last["hmm_semantic_regime"]),
                    "turbulence":   float(last["turbulence_score"]),
                    "oof_pred":     float(last["oof_pred"]),
                    "meta_prob":    float(meta_prob),
                    "veto_reason":  str(veto_reason),
                    "close":        float(last["close"]),
                    "ts":           time.time(),
                    "n_bars":       len(buffer_copy),
                    "is_event":     is_event,
                    "atr14":        atr14,
                    "htf_regime":   htf_regime,
                })

            # 8. Dashboard Diagnostics (fire-and-forget, never blocks pipeline)
            try:
                def _sf(v, default=None):
                    """Safe float: return None instead of NaN/Inf so JSON stays valid."""
                    try:
                        f = float(v)
                        return f if np.isfinite(f) else default
                    except Exception:
                        return default

                _thr = _sf(_thr_val)   # already computed above
                self._diag_queue.put_nowait({
                    "target_pos":           _sf(self._current_target_pos, 0.0),
                    "raw_kelly":            _sf(raw_target_pos, 0.0),
                    "regime":               str(last["hmm_semantic_regime"]),
                    "pending_regime":           self._pending_regime,
                    "pending_regime_count":     self._pending_regime_count,
                    "pending_regime_threshold": _hysteresis_needed if self._pending_regime else None,
                    "turbulence":           _sf(last["turbulence_score"], 0.0),
                    "turbulence_thr":       _thr,
                    "oof_pred":             _sf(last["oof_pred"], 0.5),
                    "close":                _sf(last["close"]),
                    "ts":                   time.time(),
                    "n_bars":               len(buffer_copy),
                    "is_event":             is_event,
                    "cusum_pos_sum":        _sf(self._cusum._pos_sum, 0.0),
                    "cusum_neg_sum":        _sf(self._cusum._neg_sum, 0.0),
                    "cusum_threshold":      _sf(cusum_h, 0.0),
                    "hmm_fitted":           self._hmm_fitted,
                    "bars_since_fit":       self._bars_since_fit,
                    "pipeline_inflight":    False,  # published at end of successful run
                    "alpha_bypass":         _bypass_applied,
                    "bypass_hold_bars":     self._bypass_hold_bars,
                    "atr14":                _sf(atr14),
                    "confidence_scale":     _sf(confidence_scale, 0.0),
                    "htf_regime":           htf_regime,
                    "htf_multiplier":       _sf(htf_mult, 0.5),
                    "htf_multiplier":       _sf(htf_mult, 0.5),
                    "book_imbalance":       _sf(_bar_imbalance, 0.0),
                    "meta_prob":            _sf(meta_prob, 0.0),
                    "veto_reason":          str(veto_reason),
                    "expected_cost_bps":    _sf(expected_cost_bps, 0.0),
                })
            except Exception:
                pass  # Never let diagnostics crash the pipeline

            logger.info(
                "Pipeline OK | bar=%d close=%.2f alpha=%.3f regime=%s target_pos=%.4f (event=%s)",
                len(buffer_copy),
                float(last["close"]),
                float(last["oof_pred"]),
                str(last["hmm_semantic_regime"]),
                float(self._current_target_pos),
                "YES" if is_event else "no",
            )
        except Exception as e:
            logger.error("Unexpected error in pipeline worker: %s", e, exc_info=True)
        finally:
            self._pipeline_inflight = False


    def run(self):
        if not _ZMQ_AVAILABLE:
            logger.error("pyzmq not installed.  ZMQ listener inactive.")
            return

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self._zmq_addr)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC_DOLLAR_BAR)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC_BOOK_TICKER)
        # Non-blocking poll so we can check _running
        sock.setsockopt(zmq.RCVTIMEO, 1000)

        diag_ctx = zmq.Context()
        diag_sock = diag_ctx.socket(zmq.PUB)
        diag_sock.setsockopt(zmq.SNDHWM, 10)   # drop if no subscriber, never block
        diag_sock.bind(ZMQ_DIAG_ADDR)

        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info("ZMQ listener connected to %s | DIAG PUB at %s", self._zmq_addr, ZMQ_DIAG_ADDR)

        while self._running:
            try:
                # BUG #23 FIX: Safety reset for inflight flag if stuck for >60s
                if self._pipeline_inflight and (time.time() - self._inflight_since > 60):
                    logger.warning("[SAFETY] Resetting stuck pipeline_inflight flag.")
                    self._pipeline_inflight = False

                topic, payload_bytes = sock.recv_multipart()
                payload = json.loads(payload_bytes.decode())

                if topic == TOPIC_DOLLAR_BAR:
                    self._bar_buffer.append(payload)
                    if len(self._bar_buffer) > 4000:
                        self._bar_buffer = self._bar_buffer[-2000:]

                    # Update n_bars immediately so tests can track progress
                    with self._lock:
                        self._signal["n_bars"] = len(self._bar_buffer)

                    # BUG #23 FIX: Coalescing — skip if a pipeline is already inflight.
                    # This prevents the queue from growing during long HMM refits.
                    if not self._pipeline_inflight:
                        self._pipeline_inflight = True
                        self._inflight_since = time.time()
                        try:
                            self._executor.submit(self._run_pipeline)
                        except Exception as e:
                            self._pipeline_inflight = False
                            logger.warning("Failed to submit pipeline: %s", e)

                elif topic == TOPIC_BOOK_TICKER:
                    with self._lock:
                        self._signal["best_bid"]       = payload.get("bid")
                        self._signal["best_ask"]       = payload.get("ask")
                        self._signal["mid"]            = (payload.get("bid", 0) + payload.get("ask", 0)) / 2 if payload.get("bid") else 0
                        self._signal["book_imbalance"] = payload.get("l1_imb", 0.0)

            except zmq.Again:
                pass  # Timeout — just loop
            except Exception as e:
                logger.error("ZMQ recv error: %s", e)
                time.sleep(1)

            # Drain diagnostics queue and publish (fire-and-forget, never blocks)
            while not self._diag_queue.empty():
                try:
                    diag = self._diag_queue.get_nowait()
                    diag_sock.send_multipart(
                        [b"DIAGNOSTICS", json.dumps(diag, default=str).encode()],
                        flags=zmq.NOBLOCK,
                    )
                except Exception:
                    pass

        sock.close()
        ctx.term()
        diag_sock.close()
        diag_ctx.term()
        logger.info("ZMQ listener stopped.")


class InstitutionalDollarStrategy(IStrategy):
    """
    Freqtrade execution shell backed by the institutional 7-layer pipeline.
    Receives completed Dollar Bars + book state from market_data_daemon.py
    via ZeroMQ PUB/SUB.  All intelligence runs in a background thread.

    Run with:
        nohup python RL_Trading/services/market_data_daemon.py &
        freqtrade trade -s InstitutionalDollarStrategy --dry-run
    """

    INTERFACE_VERSION     = 3
    timeframe             = "1m"
    startup_candle_count  = 1  # Reduced from 100 for faster boot (strategy uses ZMQ)
    process_only_new_candles = True

    # Mute all native Freqtrade risk controls — the pipeline governs these
    minimal_roi          = {"0": 100}
    stoploss             = -0.99       # hard floor; custom_stoploss handles normal exits
    trailing_stop        = False
    use_exit_signal      = True
    use_custom_stoploss  = True

    # BUG #21 FIX: Moved mutable state to instance level in bot_start
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_model      = None
        self._turbulence       = None
        self._hmm              = None
        self._sizer            = None
        self._zmq_listener     = None
        self._signal_lock      = threading.Lock()
        self._latest_signal    = {
            "target_pos": 0.0, "regime": "unknown", "turbulence": 0.0,
            "oof_pred": 0.5, "close": None, "ts": 0.0,
            "best_bid": None, "best_ask": None, "mid": None, "n_bars": 0,
        }
        self._boot_time = time.time()
        self._last_watchdog_log_time = 0.0

    V_BAR_TARGET = 2_000_000.0

    def _seed_buffer_from_cache(self) -> None:
        """
        Pre-populate the bar buffer from the cached feather file so the pipeline
        fires immediately after restart instead of waiting for 60+ new bars.
        """
        cache_path = _RL_DIR / "cache" / "dollar_bars_btc_2000000_regimes.feather"
        if not cache_path.exists():
            logger.info("Cache not found at %s — buffer will fill from live stream.", cache_path)
            return
        try:
            import pyarrow.feather as feather
            df = feather.read_feather(str(cache_path))
            
            # BUG #18 FIX: Recency validation using internal bar timestamp (t_close)
            if "t_close" in df.columns:
                last_ts = df["t_close"].iloc[-1]
                if last_ts > 1e11: last_ts /= 1000.0
                age = time.time() - last_ts
                # Historical bars are valid for turbulence baseline even if old.
                # Only skip if data is absurdly stale (>30 days).
                if age > 86400 * 30:
                    logger.warning("Cache data is too stale (%.0fd old) — skipping seeding.", age / 86400)
                    return
                logger.info("Cache age: %.1fh", age / 3600)

            # Seed 2000 bars so MahalanobisTurbulence (window=1000) has enough
            # history to compute real distances on the first pipeline run.
            df = df.tail(2000).copy()

            # Derive t_open/t_close (unix milliseconds) from the feather's
            # datetime columns so the HTF aggregation can build 1h buckets
            # across the full seeded history.
            if "end_time" in df.columns and "t_close" not in df.columns:
                df["t_close"] = (pd.to_datetime(df["end_time"]).astype("int64") // 1_000_000)
            if "start_time" in df.columns and "t_open" not in df.columns:
                df["t_open"] = (pd.to_datetime(df["start_time"]).astype("int64") // 1_000_000)

            # D-10 FIX: Include microstructure columns when present in the feather.
            priority_cols = [
                "t_open", "t_close",
                "open", "high", "low", "close", "volume",
                "buy_volume", "sell_volume", "aggressor_ratio",
                "notional", "trade_count", "cvd",
                "intraday_range_feature", "log_return_feature", "volatility_24_feature",
            ]
            existing = [c for c in priority_cols if c in df.columns]
            records = df[existing].to_dict(orient="records")
            self._zmq_listener._bar_buffer = records

            logger.info("Buffer seeded with %d bars from cache (%s)", len(records), cache_path.name)
        except Exception as e:
            logger.warning("Cache seeding failed: %s", e)


    def bot_start(self, **kwargs) -> None:
        logger.info("[BASELINE-V1] Booting Institutional Hardened Candidate (Shadow-Live mode)...")

        # 1. Load Frozen Assets from Deployment Path
        deploy_path = _RL_DIR / "deployments" / "baseline_hardened_v1"
        model_path = deploy_path / "alpha_model.txt"
        meta_path  = deploy_path / "gatekeeper.txt"
        
        self._alpha_model = None
        if model_path.exists():
            self._alpha_model = lgb.Booster(model_file=str(model_path))
            logger.info("[BASELINE-V1] Alpha Model loaded from deployment: %s", model_path)
        else:
            logger.error("[BASELINE-V1] CRITICAL: Alpha model missing in %s", model_path)

        self._meta_model = None
        if meta_path.exists():
            self._meta_model = lgb.Booster(model_file=str(meta_path))
            logger.info("[BASELINE-V1] Meta-Gatekeeper loaded from deployment: %s", meta_path)
        else:
            logger.error("[BASELINE-V1] CRITICAL: Meta-model missing in %s", meta_path)

        # 2. Instantiate Risk Engines
        self._turbulence = MahalanobisTurbulence(window=1000, step=250)
        self._hmm        = HMMRegimeModel(n_components=3, n_init=3)
        self._hmm_htf    = HMMRegimeModel(n_components=3, n_init=3)
        self._sizer      = FractionalKellySizer(kelly_fraction=0.5, max_drawdown=0.10)

        # 3. Start background ZMQ listener thread
        self._zmq_listener = _ZmqListener(
            alpha_model=self._alpha_model,
            meta_model=self._meta_model,
            alpha_slow_model=None, # Baseline V1 uses model_a only
            turbulence_engine=self._turbulence,
            hmm_model=self._hmm,
            hmm_model_htf=self._hmm_htf,
            sizer=self._sizer,
            signal_store=self._latest_signal,
            lock=self._signal_lock,
        )

        # 4. Pre-populate buffer SO seeding happens AFTER listener exists but BEFORE it starts
        self._seed_buffer_from_cache()

        # 5. Kick off the thread
        self._zmq_listener.start()
        logger.info("ZMQ listener thread started (daemon=%s)", ZMQ_DAEMON_ADDR)

    def bot_stopped(self) -> None:
        if self._zmq_listener:
            self._zmq_listener.stop()
            self._zmq_listener.join(timeout=3)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Thin shell: reads latest signal from the ZMQ listener thread and stamps
        it onto every row of the 1m Freqtrade candle frame.
        No computation happens here — just state propagation.
        """
        with self._signal_lock:
            sig = dict(self._latest_signal)  # snapshot

        # D-06 FIX: Reconcile virtual position with Freqtrade's actual fill state.
        # The listener tracks _current_target_pos independently from real fills.
        # Partial fills, rejections, or slippage create divergence over time.
        # Here we read the real exposure and sync it back to the listener.
        try:
            self._reconcile_actual_position()
        except Exception as e:
            logger.debug("[D-06] Position reconciliation skipped: %s", e)

        # DESIGN-04 FIX: Watchdog — alert if signal is stale (ZMQ listener may be dead)
        sig_age = time.time() - sig.get("ts", 0)
        now = time.time()
        
        # 1. Warmup / Boot timeout
        if sig.get("n_bars", 0) == 0 and (now - self._boot_time > 600):
            if now - self._last_watchdog_log_time > 300:
                logger.error("[WATCHDOG] No bars received after 10m. Daemon/ZMQ issue?")
                self._last_watchdog_log_time = now
            # BUG #20 FIX: Ensure columns exist before early return
            for col, default in [("target_pos", 0.0), ("regime", "unknown"),
                                 ("turbulence", 0.0), ("oof_pred", 0.5), ("n_bars", 0)]:
                dataframe[col] = default
            return dataframe

        # 2. Stale signal timeout
        if sig_age > 1800 and sig.get("n_bars", 0) > 0:
            if now - self._last_watchdog_log_time > 300:
                logger.error(
                    "[WATCHDOG] Signal is %.0fs old. ZMQ listener thread may be dead. "
                    "FORCING target_pos=0 for safety.",
                    sig_age
                )
                self._last_watchdog_log_time = now

            # BUG #20 FIX: Emergency flat in BOTH locations (local copy and shared state)
            sig["target_pos"] = 0.0
            dataframe["target_pos"] = 0.0
            with self._signal_lock:
                self._latest_signal["target_pos"] = 0.0
                # BUG-Watchdog FIX: Do NOT refresh ts here to allow sig_age to persist

        dataframe["target_pos"]  = sig.get("target_pos",  0.0)
        dataframe["regime"]      = sig.get("regime",      "unknown")
        dataframe["turbulence"]  = sig.get("turbulence",  0.0)
        dataframe["oof_pred"]    = sig.get("oof_pred",    0.5)
        dataframe["n_bars"]      = sig.get("n_bars",      0)

        return dataframe


    def _reconcile_actual_position(self) -> None:
        """
        D-06: Sync _current_target_pos in the ZMQ listener with the real
        fill state tracked by Freqtrade's portfolio manager.

        Logic:
          - Read open trades for the current pair.
          - Compute actual exposure = open_trade_value / max_stake.
          - If actual exposure differs from the listener's virtual position by
            more than 3%, update the listener so hysteresis operates correctly.

        This runs every ~1m (each populate_indicators cycle) — lightweight.
        """
        if not self._zmq_listener:
            return

        try:
            from freqtrade.persistence import Trade
            open_trades = Trade.get_open_trades()
        except Exception:
            return  # Not available in dry-run initial boot or test mode

        if not open_trades:
            actual_exposure = 0.0
        else:
            # Mark-to-market reconciliation (current price)
            total_value = sum(
                t.amount * t.current_rate for t in open_trades
            )
            try:
                max_stake = self.wallets.get_available_capital() + total_value
                actual_exposure = total_value / max_stake if max_stake > 0 else 0.0
            except Exception:
                return  # wallets not yet initialised

        virtual_pos = self._zmq_listener._current_target_pos
        drift = abs(actual_exposure - virtual_pos)

        if drift > 0.03:  # 3% reconciliation threshold
            logger.info(
                "[D-06] Position reconciliation: virtual=%.3f actual=%.3f drift=%.3f → syncing",
                virtual_pos, actual_exposure, drift
            )
            self._zmq_listener.set_virtual_position(actual_exposure)


    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Harmonized Entry Gate: Only enter if pipeline has committed target >= 10%
        dataframe["enter_long"] = 0
        dataframe.loc[
            (dataframe["target_pos"] >= 0.10) & 
            (dataframe["regime"] != "unknown"),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe.loc[dataframe["target_pos"] <= 0.0, "exit_long"] = 1
        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        Institutional Exit Audit: distinguishes between Alpha reversal and Meta-veto.
        """
        with self._signal_lock:
            targ        = self._latest_signal.get("target_pos", 0.0)
            veto_reason = self._latest_signal.get("veto_reason", "none")
            alpha_prob  = self._latest_signal.get("oof_pred", 0.5)
            meta_prob   = self._latest_signal.get("meta_prob", 0.0)

        if targ <= 0:
            if veto_reason != "none":
                logger.info("[EXIT-AUDIT] Meta-Veto exit: %s (meta_prob=%.3f)", veto_reason, meta_prob)
                return f"meta_veto_{veto_reason}"
            
            if alpha_prob < 0.50:
                logger.info("[EXIT-AUDIT] Alpha reversal exit (prob=%.3f)", alpha_prob)
                return "alpha_reversal"
            
            # Fallback
            return "pipeline_exit_zero"
        
        return None

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Sets stake as a fraction of max_stake based on the current Kelly target."""
        with self._signal_lock:
            targ = self._latest_signal.get("target_pos", 0.0)

        if targ <= 0:
            return 0.0

        # Hard floor: do not open ultra-small positions
        stake = max_stake * targ
        if min_stake and stake < min_stake:
            return 0.0

        return stake

    def custom_entry_price(self, pair: str, current_time, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Passive execution: anchor to best_bid for buys, best_ask for sells.
        Falls back to proposed_rate if no live book data is available.
        """
        with self._signal_lock:
            best_bid = self._latest_signal.get("best_bid")
            best_ask = self._latest_signal.get("best_ask")

        if side == "buy" and best_bid and best_bid > 0:
            return float(best_bid)
        if side == "sell" and best_ask and best_ask > 0:
            return float(best_ask)

        return proposed_rate

    def custom_stoploss(self, pair: str, trade, current_time,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        T2.4: ATR-based trailing stop.
        Trails 2.5× ATR14 below the current price on every tick.
        Freqtrade keeps the highest stop seen, so this naturally trails upward.
        Falls back to the hard stoploss (-0.99) when ATR is unavailable.
        """
        with self._signal_lock:
            atr14 = self._latest_signal.get("atr14", None)

        if not atr14 or atr14 <= 0 or current_rate <= 0:
            return self.stoploss

        stop_pct = (2.5 * atr14) / current_rate
        # Floor at 0.5% to avoid premature stop on ultra-low-vol bars;
        # cap at 10% to prevent excessively wide stops in panic conditions.
        return -float(np.clip(stop_pct, 0.005, 0.10))
