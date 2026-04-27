#!/usr/bin/env python3
"""
scripts/pipeline_replay.py
───────────────────────────
Offline replay of historical dollar bars through the full institutional pipeline.

Runs the EXACT same logic as _ZmqListener._run_pipeline() but on cached data,
so you can validate changes in minutes instead of waiting days for live markets.

Usage:
    conda run -n freqtrade python scripts/pipeline_replay.py [OPTIONS]

Options:
    --bars N          Number of bars to replay (default: all)
    --start-bar N     Start from bar N (default: 2000, for warmup)
    --window N        Rolling buffer size (default: 2000)
    --step N          Process every Nth bar (default: 1, use 5-10 for speed)
    --output FILE     Save results CSV (default: reports/replay_results.csv)
    --compare         Run old vs new HTF logic side-by-side
    --verbose         Print every pipeline tick
    --plot            Generate HTML chart of decisions

Example:
    # Quick validation (every 5th bar, last 10k bars):
    python scripts/pipeline_replay.py --bars 10000 --step 5

    # Full comparison run:
    python scripts/pipeline_replay.py --compare --output reports/replay_compare.csv
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_DIR = _SCRIPT_DIR.parent
for _p in [str(_RL_DIR), str(_RL_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lightgbm as lgb
from utils.signal_features import build_feature_matrix
from utils.risk_directors import MahalanobisTurbulence, HMMRegimeModel
from utils.position_sizer import FractionalKellySizer
from utils.filters import SymmetricCUSUMFilter


# ── HTF Config (must match InstitutionalDollarStrategy.py) ──────────────────
_HTF_CONFIG_NEW = {
    "bull_calm":        (1.00, 0.00),
    "bull_neutral":     (0.80, 0.00),
    "high_vol_rebound": (0.60, 0.05),
    "bear_neutral":     (0.30, 0.00),
    "bear_calm":        (0.20, 0.05),
    "panic_selloff":    (0.00, 0.00),
    "unknown":          (0.50, 0.00),
}

_HTF_MULTIPLIERS_OLD = {
    "bull_calm":        1.00,
    "bull_neutral":     0.80,
    "high_vol_rebound": 0.60,
    "bear_neutral":     0.30,
    "bear_calm":        0.20,
    "panic_selloff":    0.00,
    "unknown":          0.50,
}

_BYPASS_REGIMES_OLD = {"bull_calm", "bull_neutral"}
_BYPASS_REGIMES_NEW = {"bull_calm", "bull_neutral", "high_vol_rebound"}

HYSTERESIS_TO_BULL_OLD = 3
HYSTERESIS_TO_BULL_NEW = 3   # restored: 2 was over-firing bypass into unconfirmed bear
HYSTERESIS_TO_BEAR = 6

_BULL_REGIMES = {"bull_calm", "bull_neutral", "high_vol_rebound"}


class PipelineReplay:
    """Replays dollar bars through the institutional pipeline offline."""

    def __init__(self, alpha_model, use_new_logic=True, verbose=False):
        self.alpha = alpha_model
        self.turb = MahalanobisTurbulence(window=1000, step=250)
        self.hmm = HMMRegimeModel(n_components=3, n_init=3)
        self.hmm_htf = HMMRegimeModel(n_components=3, n_init=3)
        self.sizer = FractionalKellySizer(kelly_fraction=0.5, max_drawdown=0.10)
        self.cusum = SymmetricCUSUMFilter()
        self.use_new = use_new_logic
        self.verbose = verbose

        # State
        self.hmm_fitted = False
        self.bars_since_fit = 0
        self.htf_fitted = False
        self.htf_bars_since_fit = 0
        self.current_target_pos = 0.0
        self.last_t_processed = 0.0

        # Hysteresis
        self.pending_regime = None
        self.pending_regime_count = 0
        self.committed_regime = "unknown"
        self._hysteresis_to_bull = HYSTERESIS_TO_BULL_NEW if use_new_logic else HYSTERESIS_TO_BULL_OLD

        # Bypass hold: sustain 10% floor for N bars after bypass entry
        self.bypass_hold_bars = 0

    def _apply_hysteresis(self, raw_regime: str) -> str:
        """Apply asymmetric temporal hysteresis to regime labels."""
        hysteresis_needed = (
            self._hysteresis_to_bull if raw_regime in _BULL_REGIMES
            else HYSTERESIS_TO_BEAR
        )
        if raw_regime == self.committed_regime:
            self.pending_regime = None
            self.pending_regime_count = 0
        elif raw_regime == self.pending_regime:
            self.pending_regime_count += 1
            if self.pending_regime_count >= hysteresis_needed:
                self.committed_regime = raw_regime
                self.pending_regime = None
                self.pending_regime_count = 0
        else:
            self.pending_regime = raw_regime
            self.pending_regime_count = 1

        return self.committed_regime

    def _apply_htf(self, raw_pos: float, htf_regime: str, ltf_regime: str) -> float:
        """Apply HTF filter — new (floor) or old (pure multiplication)."""
        if self.use_new:
            htf_mult, htf_floor = _HTF_CONFIG_NEW.get(htf_regime, (0.50, 0.00))
            ltf_recovery = ltf_regime in ("high_vol_rebound", "bull_calm", "bull_neutral")
            if htf_mult < 1.0:
                multiplicative = raw_pos * htf_mult
                if ltf_recovery and htf_floor > 0:
                    return max(multiplicative, min(raw_pos, htf_floor))
                return multiplicative
            return raw_pos
        else:
            htf_mult = _HTF_MULTIPLIERS_OLD.get(htf_regime, 0.50)
            return raw_pos * htf_mult if htf_mult < 1.0 else raw_pos

    def _compute_cusum(self, df: pd.DataFrame, idx: int) -> bool:
        """Check CUSUM event at position idx."""
        if idx < 1:
            return False
        prev_p = float(df.iloc[idx - 1]["close"])
        curr_p = float(df.iloc[idx]["close"])
        log_ret = np.log(curr_p / prev_p)

        sigma_series = df["log_return_feature"].iloc[:idx + 1].tail(100)
        sigma = sigma_series.std()

        if self.use_new:
            sigma_robust = sigma_series.abs().median() * 1.4826
            if not np.isnan(sigma_robust) and sigma_robust > 0:
                sigma_eff = min(sigma, sigma_robust * 2.0)
            else:
                sigma_eff = sigma
        else:
            sigma_eff = sigma

        cusum_h = 3.5 * sigma_eff if (not np.isnan(sigma_eff) and sigma_eff > 0) else 0.005
        return self.cusum.check(log_ret, cusum_h)

    def run_tick(self, df_buffer: pd.DataFrame) -> dict:
        """Run one pipeline tick on the buffer. Returns decision dict."""
        if len(df_buffer) < 20:
            return None

        # Feature engineering
        try:
            X = build_feature_matrix(df_buffer.copy(), eth_df=None, funding_series=None)
        except Exception:
            return None

        if len(X) < 20:
            return None

        model_features = self.alpha.feature_name()
        for feat in model_features:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[model_features].fillna(0.0)

        # Alpha
        preds = self.alpha.predict(X)
        df_buffer["oof_pred"] = pd.Series(preds).fillna(0.5).clip(0.0, 1.0).values[:len(df_buffer)]

        # Turbulence
        risk_vec = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
        available_risk = [c for c in risk_vec if c in X.columns]
        if len(available_risk) < 2:
            return None

        for col in available_risk + ["log_return_feature", "volatility_24_feature"]:
            if col in X.columns:
                df_buffer[col] = X[col].values[:len(df_buffer)]

        turb_series = self.turb.compute(df_buffer, available_risk)
        if turb_series is not None:
            turb_p95 = turb_series.quantile(0.95) if turb_series.notna().sum() > 10 else 5.0
            turb_p95 = float(turb_p95) if not np.isnan(turb_p95) else 5.0
            df_buffer["turbulence_score"] = turb_series.fillna(turb_p95)
        else:
            df_buffer["turbulence_score"] = 5.0

        adaptive_thr = self.turb.rolling_threshold(df_buffer["turbulence_score"])

        # HMM LTF
        hmm_feats = ["log_return_feature", "volatility_24_feature"]
        extended_feats = [
            "log_return_feature", "volatility_24_feature",
            "aggressor_ratio", "intraday_range_feature",
        ]
        hmm_feats = [f for f in extended_feats if f in df_buffer.columns] or hmm_feats
        available_hmm = [c for c in hmm_feats if c in df_buffer.columns]

        try:
            if not self.hmm_fitted or self.bars_since_fit >= 500:
                self.hmm.fit(df_buffer, available_hmm)
                self.hmm_fitted = True
                self.bars_since_fit = 0

            assert self.hmm.features[0] == "log_return_feature"
            current_regime_raw = self.hmm.predict_current(df_buffer.tail(50))
            current_regime = self._apply_hysteresis(current_regime_raw)
            df_buffer["hmm_semantic_regime"] = current_regime
            self.bars_since_fit += 1
        except Exception:
            current_regime = "unknown"
            df_buffer["hmm_semantic_regime"] = "unknown"

        last = df_buffer.iloc[-1]

        # Kelly sizing
        raw_target_pos = self.sizer.size_portfolio(
            probabilities=df_buffer["oof_pred"],
            regimes=df_buffer["hmm_semantic_regime"],
            turbulence=df_buffer["turbulence_score"],
            adaptive_threshold=adaptive_thr,
            risk_scales=df_buffer["volatility_24_feature"],
            dof=len(available_risk),
        ).iloc[-1]

        # Confidence scaling
        last_oof = float(last["oof_pred"])
        confidence_scale = min(1.0, (2.0 * abs(last_oof - 0.5)) / 0.15)
        raw_target_pos = float(raw_target_pos) * confidence_scale

        # HTF
        htf_regime = "unknown"
        try:
            if "end_time" in df_buffer.columns:
                ts_col = pd.to_datetime(df_buffer["end_time"])
            elif "t_close" in df_buffer.columns:
                ts_col = pd.to_datetime(df_buffer["t_close"], unit="ms", utc=True)
            else:
                ts_col = None

            if ts_col is not None:
                df_htf = (
                    df_buffer.assign(_hour=ts_col.dt.floor("1h"))
                    .groupby("_hour", as_index=False)
                    .agg(
                        open=("open", "first"),
                        high=("high", "max"),
                        low=("low", "min"),
                        close=("close", "last"),
                        volume=("volume", "sum"),
                        log_return_feature=("log_return_feature", "sum"),
                        volatility_24_feature=("volatility_24_feature", "mean"),
                        intraday_range_feature=("intraday_range_feature", "mean"),
                    )
                    .reset_index(drop=True)
                )
                if "aggressor_ratio" in df_buffer.columns:
                    df_htf_aggr = (
                        df_buffer.assign(_hour=ts_col.dt.floor("1h"))
                        .groupby("_hour", as_index=False)
                        .agg(aggressor_ratio=("aggressor_ratio", "mean"))
                    )
                    df_htf["aggressor_ratio"] = df_htf_aggr["aggressor_ratio"].values

                if len(df_htf) >= 20:
                    htf_feats = [c for c in [
                        "log_return_feature", "volatility_24_feature",
                        "aggressor_ratio", "intraday_range_feature",
                    ] if c in df_htf.columns]

                    if not self.htf_fitted or self.htf_bars_since_fit >= 50:
                        self.hmm_htf.fit(df_htf, htf_feats)
                        self.htf_fitted = True
                        self.htf_bars_since_fit = 0

                    assert self.hmm_htf.features[0] == "log_return_feature"
                    htf_regime = self.hmm_htf.predict_current(df_htf.tail(6))
                    self.htf_bars_since_fit += 1
        except Exception:
            pass

        pos_before_htf = raw_target_pos
        raw_target_pos = self._apply_htf(raw_target_pos, htf_regime, current_regime)

        # CUSUM
        is_event = self._compute_cusum(df_buffer, len(df_buffer) - 1)

        # Alpha bypass
        bypass_applied = False
        _thr_val = None
        if adaptive_thr is not None and adaptive_thr.notna().any():
            try:
                _v = float(adaptive_thr.iloc[-1])
                if np.isfinite(_v):
                    _thr_val = _v
            except Exception:
                pass

        _turb_last = float(last["turbulence_score"])
        _turb_ok = (_thr_val is None) or (_turb_last < _thr_val)
        _bar_imbalance = float(last.get("book_imbalance", 0.0))
        _imbalance_ok = _bar_imbalance >= -0.20

        # Bypass hold: sustain 10% floor on bars following a bypass entry
        if self.use_new and self.bypass_hold_bars > 0:
            _bull_targets = {"bull_calm", "bull_neutral", "high_vol_rebound"}
            _hold_regime_ok = (
                self.pending_regime in _bull_targets
                or self.committed_regime in _bull_targets
            )
            _hold_alpha_ok = float(last["oof_pred"]) >= 0.45
            if _hold_regime_ok and _hold_alpha_ok:
                if raw_target_pos < 0.10:
                    raw_target_pos = 0.10
                self.bypass_hold_bars -= 1
            else:
                self.bypass_hold_bars = 0

        bypass_regimes = _BYPASS_REGIMES_NEW if self.use_new else _BYPASS_REGIMES_OLD
        if (is_event
                and float(last["oof_pred"]) >= 0.65
                and _turb_ok
                and _imbalance_ok
                and self.pending_regime in bypass_regimes
                and raw_target_pos < 0.10):
            raw_target_pos = 0.10
            bypass_applied = True
            if self.use_new:
                self.bypass_hold_bars = self._hysteresis_to_bull

        # Entry gate
        pos_diff = abs(raw_target_pos - self.current_target_pos)
        should_update = False
        if is_event:
            if self.current_target_pos == 0:
                if raw_target_pos >= 0.10:
                    should_update = True
            elif raw_target_pos < self.current_target_pos:
                if (pos_diff >= 0.05) or (raw_target_pos == 0):
                    should_update = True
            elif raw_target_pos > self.current_target_pos:
                if pos_diff >= 0.10:
                    should_update = True

        if should_update:
            self.current_target_pos = raw_target_pos

        return {
            "close": float(last["close"]),
            "alpha": float(last["oof_pred"]),
            "regime_raw": current_regime if 'current_regime_raw' not in dir() else current_regime_raw,
            "regime_committed": current_regime,
            "pending_regime": self.pending_regime,
            "htf_regime": htf_regime,
            "kelly_raw": pos_before_htf,
            "kelly_after_htf": raw_target_pos,
            "target_pos": self.current_target_pos,
            "confidence": confidence_scale,
            "is_event": is_event,
            "bypass": bypass_applied,
            "bypass_hold_bars": self.bypass_hold_bars,
            "should_update": should_update,
            "turbulence": _turb_last,
        }


def load_data(feather_path: str, n_bars: int = None, start_bar: int = 0) -> pd.DataFrame:
    """Load dollar bars from feather file."""
    df = pd.read_feather(feather_path)

    # Rename columns to match pipeline expectations
    renames = {}
    if "buy_vol" in df.columns and "buy_volume" not in df.columns:
        renames["buy_vol"] = "buy_volume"
    if "sell_vol" in df.columns and "sell_volume" not in df.columns:
        renames["sell_vol"] = "sell_volume"
    if renames:
        df = df.rename(columns=renames)

    if start_bar > 0:
        df = df.iloc[start_bar:].reset_index(drop=True)
    if n_bars and n_bars < len(df):
        df = df.iloc[:n_bars].reset_index(drop=True)

    return df


def replay(df: pd.DataFrame, alpha_model, window: int = 2000, step: int = 1,
           use_new_logic: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Replay bars through the pipeline, returning decision log."""

    pipeline = PipelineReplay(alpha_model, use_new_logic=use_new_logic, verbose=verbose)
    results = []
    total = len(df) - window
    t0 = time.time()

    print(f"Replaying {total // step} ticks (window={window}, step={step}, "
          f"logic={'NEW' if use_new_logic else 'OLD'})...")

    for i in range(window, len(df), step):
        buf = df.iloc[i - window:i].copy().reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tick = pipeline.run_tick(buf)

        if tick is None:
            continue

        tick["bar_idx"] = i
        if "end_time" in df.columns:
            tick["timestamp"] = str(df.iloc[i - 1]["end_time"])
        results.append(tick)

        if verbose and tick["should_update"]:
            print(f"  [{i:>6d}] TRADE: pos={tick['target_pos']:.3f} "
                  f"regime={tick['regime_committed']} htf={tick['htf_regime']} "
                  f"alpha={tick['alpha']:.3f} bypass={tick['bypass']}")

        if i % (1000 * step) == 0:
            elapsed = time.time() - t0
            pct = (i - window) / max(total, 1) * 100
            rate = (i - window) / max(elapsed, 0.01)
            eta = (total - (i - window)) / max(rate, 0.01)
            print(f"  Progress: {pct:.1f}% ({i - window}/{total}) "
                  f"| {rate:.0f} bars/s | ETA: {eta:.0f}s")

    return pd.DataFrame(results)


def print_summary(results: pd.DataFrame, label: str = ""):
    """Print a summary of the replay results."""
    print(f"\n{'='*60}")
    print(f"REPLAY SUMMARY {label}")
    print(f"{'='*60}")

    total_ticks = len(results)
    print(f"Total ticks processed: {total_ticks:,}")

    if total_ticks == 0:
        print("No data to summarize.")
        return

    # Trades
    trades = results[results["should_update"]]
    entries = trades[trades["target_pos"] > 0]
    exits = trades[trades["target_pos"] == 0]
    print(f"\nTrades: {len(trades)} total ({len(entries)} entries, {len(exits)} exits)")

    # Bypass activations
    bypasses = results[results["bypass"]]
    print(f"Alpha Bypass activations: {len(bypasses)}")

    # CUSUM events
    events = results[results["is_event"]]
    print(f"CUSUM events: {len(events)} ({len(events)/max(total_ticks,1)*100:.1f}%)")

    # Regime distribution
    print(f"\nRegime distribution (committed):")
    regime_counts = results["regime_committed"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / total_ticks * 100
        print(f"  {regime:25s}: {count:>6,} ({pct:5.1f}%)")

    # HTF regime distribution
    print(f"\nHTF Regime distribution:")
    htf_counts = results["htf_regime"].value_counts()
    for regime, count in htf_counts.items():
        pct = count / total_ticks * 100
        print(f"  {regime:25s}: {count:>6,} ({pct:5.1f}%)")

    # Alpha stats
    print(f"\nAlpha (oof_pred) stats:")
    print(f"  Mean: {results['alpha'].mean():.4f}")
    print(f"  >0.55: {(results['alpha'] > 0.55).sum():,} ({(results['alpha'] > 0.55).mean()*100:.1f}%)")
    print(f"  >0.60: {(results['alpha'] > 0.60).sum():,} ({(results['alpha'] > 0.60).mean()*100:.1f}%)")
    print(f"  >0.65: {(results['alpha'] > 0.65).sum():,} ({(results['alpha'] > 0.65).mean()*100:.1f}%)")

    # Position stats
    positioned = results[results["target_pos"] > 0]
    print(f"\nTime in position: {len(positioned):,} ticks ({len(positioned)/max(total_ticks,1)*100:.1f}%)")
    if len(positioned) > 0:
        print(f"  Avg position size: {positioned['target_pos'].mean():.3f}")
        print(f"  Max position size: {positioned['target_pos'].max():.3f}")

    # Trade detail
    if len(trades) > 0:
        print(f"\nTrade details:")
        cols = ["bar_idx", "timestamp", "close", "alpha", "regime_committed",
                "htf_regime", "target_pos", "bypass"]
        available = [c for c in cols if c in trades.columns]
        print(trades[available].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Offline pipeline replay")
    parser.add_argument("--bars", type=int, default=0, help="Number of bars (0=all)")
    parser.add_argument("--start-bar", type=int, default=0, help="Start from bar N")
    parser.add_argument("--window", type=int, default=2000, help="Buffer window size")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth bar")
    parser.add_argument("--output", type=str, default="reports/replay_results.csv")
    parser.add_argument("--compare", action="store_true", help="Run old vs new side-by-side")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--data", type=str,
                        default=str(_RL_DIR / "cache" / "dollar_bars_btc_2000000_regimes.feather"))
    args = parser.parse_args()

    # Load model
    model_path = _RL_DIR / "models" / "dollar_alpha_v1" / "latest_model.txt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    alpha = lgb.Booster(model_file=str(model_path))
    print(f"Alpha model loaded: {model_path}")

    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data, n_bars=args.bars or None, start_bar=args.start_bar)
    print(f"Loaded {len(df):,} bars")

    if args.compare:
        # Run both old and new logic
        print("\n" + "="*60)
        print("RUNNING OLD LOGIC")
        print("="*60)
        results_old = replay(df, alpha, window=args.window, step=args.step,
                             use_new_logic=False, verbose=args.verbose)
        print_summary(results_old, "[OLD LOGIC]")

        print("\n" + "="*60)
        print("RUNNING NEW LOGIC")
        print("="*60)
        results_new = replay(df, alpha, window=args.window, step=args.step,
                             use_new_logic=True, verbose=args.verbose)
        print_summary(results_new, "[NEW LOGIC]")

        # Comparison
        print("\n" + "="*60)
        print("COMPARISON: OLD vs NEW")
        print("="*60)
        trades_old = len(results_old[results_old["should_update"]])
        trades_new = len(results_new[results_new["should_update"]])
        bypass_old = len(results_old[results_old["bypass"]])
        bypass_new = len(results_new[results_new["bypass"]])
        events_old = len(results_old[results_old["is_event"]])
        events_new = len(results_new[results_new["is_event"]])
        pos_old = (results_old["target_pos"] > 0).sum()
        pos_new = (results_new["target_pos"] > 0).sum()

        print(f"  {'Metric':30s} {'OLD':>10s} {'NEW':>10s} {'Δ':>10s}")
        print(f"  {'-'*60}")
        print(f"  {'Trades':30s} {trades_old:>10d} {trades_new:>10d} {trades_new-trades_old:>+10d}")
        print(f"  {'Alpha Bypasses':30s} {bypass_old:>10d} {bypass_new:>10d} {bypass_new-bypass_old:>+10d}")
        print(f"  {'CUSUM Events':30s} {events_old:>10d} {events_new:>10d} {events_new-events_old:>+10d}")
        print(f"  {'Ticks in Position':30s} {pos_old:>10d} {pos_new:>10d} {pos_new-pos_old:>+10d}")

        # Save both
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_old["logic"] = "old"
        results_new["logic"] = "new"
        combined = pd.concat([results_old, results_new], ignore_index=True)
        combined.to_csv(str(out_path), index=False)
        print(f"\nResults saved to {out_path}")
    else:
        results = replay(df, alpha, window=args.window, step=args.step,
                         use_new_logic=True, verbose=args.verbose)
        print_summary(results, "[NEW LOGIC]")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(str(out_path), index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
