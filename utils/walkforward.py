"""
utils/walkforward.py
────────────────────
Fold generators for time-series walk-forward validation.

Two separate use-cases covered:

1. `oos_subwindows(df, start, end, window_days, step_days)`:
   For a model ALREADY trained on data <= `train_end`, carve multiple
   rolling TEST windows inside [start, end] (must be after train_end).
   Each fold is fully OOS to the model — cheap "how stable is my
   point-estimate" answer. Returns list of (fold_idx, slice_df, meta).

2. `anchored_walkforward(df, anchor_start, first_test_start, val_days,
    test_days, step_days)`:
   Full walk-forward with growing train window and rolling val/test.
   Each fold yields (train_df, val_df, test_df, meta). The caller is
   responsible for REFITTING the model per fold — this generator only
   slices data. Use for the expensive "correct" evaluation of training
   stability across regimes.

Design notes
────────────
- All slicing uses pandas DatetimeIndex inclusive bounds.
- `meta` includes fold_idx, window bounds, market return (for quick
  regime classification: bull if mkt_ret > +10%, bear if < -10%).
- No side effects — pure data-slicing. Trainers do the model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldMeta:
    fold_idx:        int
    train_start:     Optional[pd.Timestamp]
    train_end:       Optional[pd.Timestamp]
    val_start:       Optional[pd.Timestamp]
    val_end:         Optional[pd.Timestamp]
    test_start:      pd.Timestamp
    test_end:        pd.Timestamp
    market_return:   float     # bh on TEST, percent
    regime_tag:      str       # "bull" / "bear" / "sideways"

    def to_dict(self):
        d = self.__dict__.copy()
        # stringify timestamps so JSON serializable
        for k, v in list(d.items()):
            if isinstance(v, pd.Timestamp):
                d[k] = v.strftime("%Y-%m-%d")
        return d


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _regime_tag(market_return_pct: float,
                threshold: float = 10.0) -> str:
    """Classify a window by its buy-and-hold return.
    > +threshold → bull, < -threshold → bear, else sideways.
    """
    if market_return_pct >  threshold: return "bull"
    if market_return_pct < -threshold: return "bear"
    return "sideways"


def _market_return_pct(df_slice: pd.DataFrame) -> float:
    if len(df_slice) < 2 or "close" not in df_slice.columns:
        return 0.0
    return float((df_slice["close"].iloc[-1] / df_slice["close"].iloc[0] - 1) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# USE-CASE 1: OOS SUBWINDOWS (no refit)
# ══════════════════════════════════════════════════════════════════════════════

def oos_subwindows(
    df: pd.DataFrame,
    start:        str,
    end:          str,
    window_days:  int = 90,
    step_days:    int = 30,
    min_bars:     int = 240,
) -> List[Tuple[pd.DataFrame, FoldMeta]]:
    """
    Slice [start, end] into rolling windows of `window_days`, stepping
    forward `step_days`. Useful for evaluating a frozen model across
    multiple OOS sub-periods.

    Example
    -------
    12-week windows stepping 4 weeks through 2024-07 → 2026-03:
        folds = oos_subwindows(df, "2024-07-01", "2026-03-18",
                               window_days=84, step_days=28)
        → ~20 folds, each a 3-month OOS slice.

    Returns a list of (slice_df, FoldMeta). `slice_df` is the TEST window
    for that fold. Caller is expected to already have loaded a frozen
    model and will evaluate it on each slice.
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    folds = []
    fold_idx = 0
    window_delta = pd.Timedelta(days=window_days)
    step_delta   = pd.Timedelta(days=step_days)

    cur_start = start_ts
    while cur_start + window_delta <= end_ts + pd.Timedelta(hours=1):
        cur_end = cur_start + window_delta
        sl = df.loc[cur_start:cur_end]
        if len(sl) >= min_bars:
            mkt = _market_return_pct(sl)
            meta = FoldMeta(
                fold_idx      = fold_idx,
                train_start   = None,
                train_end     = None,
                val_start     = None,
                val_end       = None,
                test_start    = sl.index[0],
                test_end      = sl.index[-1],
                market_return = mkt,
                regime_tag    = _regime_tag(mkt),
            )
            folds.append((sl.copy(), meta))
            fold_idx += 1
        cur_start = cur_start + step_delta

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# USE-CASE 2: ANCHORED WALK-FORWARD (refit each fold)
# ══════════════════════════════════════════════════════════════════════════════

def anchored_walkforward(
    df:                pd.DataFrame,
    anchor_start:      str,
    first_test_start:  str,
    val_days:          int = 90,
    test_days:         int = 90,
    step_days:         int = 90,
    max_folds:         Optional[int] = None,
    min_train_bars:    int = 5000,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FoldMeta]]:
    """
    Anchored walk-forward. TRAIN anchored at `anchor_start`, expands fold
    by fold. VAL precedes TEST by `val_days`. TEST rolls `step_days`
    forward per fold.

    Yields (train_df, val_df, test_df, FoldMeta).

    Layout per fold:
        train:  [anchor_start            ,  val_start - 1 bar]
        val:    [test_start - val_days   ,  test_start - 1 bar]
        test:   [test_start              ,  test_start + test_days]

    Parameters
    ----------
    anchor_start:     fixed earliest date of training.
    first_test_start: date of the first fold's test start. train_end of
                      fold 0 = first_test_start - 1 bar - val_days.
    val_days:         size of val slice, used BOTH for early-stop during
                      fit and for model selection.
    test_days:        size of test slice reported as OOS fold performance.
    step_days:        how much test start advances per fold (≥1).
    max_folds:        hard cap (default: run until data runs out).
    min_train_bars:   skip folds whose train slice is smaller than this.
    """
    anchor   = pd.Timestamp(anchor_start)
    test_start = pd.Timestamp(first_test_start)
    step     = pd.Timedelta(days=step_days)
    val_td   = pd.Timedelta(days=val_days)
    test_td  = pd.Timedelta(days=test_days)
    one_hour = pd.Timedelta(hours=1)

    fold_idx = 0
    while True:
        val_start  = test_start - val_td
        val_end    = test_start - one_hour
        train_end  = val_start - one_hour
        test_end   = test_start + test_td

        if test_end > df.index[-1]:
            break
        if max_folds is not None and fold_idx >= max_folds:
            break

        train_df = df.loc[anchor:train_end]
        val_df   = df.loc[val_start:val_end]
        test_df  = df.loc[test_start:test_end]

        if len(train_df) >= min_train_bars and len(val_df) > 0 and len(test_df) > 0:
            mkt  = _market_return_pct(test_df)
            meta = FoldMeta(
                fold_idx      = fold_idx,
                train_start   = train_df.index[0],
                train_end     = train_df.index[-1],
                val_start     = val_df.index[0],
                val_end       = val_df.index[-1],
                test_start    = test_df.index[0],
                test_end      = test_df.index[-1],
                market_return = mkt,
                regime_tag    = _regime_tag(mkt),
            )
            yield train_df, val_df, test_df, meta
            fold_idx += 1

        test_start = test_start + step


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY / AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def summarise_folds(rows: List[dict]) -> dict:
    """Take per-fold metric dicts and produce an aggregated summary.

    Input rows expected to have at least:
        fold_idx, regime_tag, portfolio_return, market_return, alpha,
        sharpe, max_dd
    """
    if not rows:
        return {}
    import statistics as stats
    keys = ["portfolio_return", "market_return", "alpha", "sharpe", "max_dd"]
    out = {"n_folds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals: continue
        out[f"{k}_mean"] = float(stats.fmean(vals))
        out[f"{k}_std"]  = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
        out[f"{k}_min"]  = float(min(vals))
        out[f"{k}_max"]  = float(max(vals))
    # Alpha consistency = fraction of folds with positive alpha
    if "alpha" in rows[0]:
        pos = sum(1 for r in rows if r["alpha"] > 0)
        out["alpha_consistency"] = pos / len(rows)
    # Per-regime breakdown
    for tag in ("bull", "bear", "sideways"):
        tag_rows = [r for r in rows if r.get("regime_tag") == tag]
        if tag_rows:
            alphas = [r["alpha"] for r in tag_rows if "alpha" in r]
            out[f"alpha_{tag}_mean"] = float(stats.fmean(alphas)) if alphas else 0.0
            out[f"n_folds_{tag}"]    = len(tag_rows)
    return out
