"""
Walk-forward fractal Fibonacci features for the RL agent.

5 features per bar, computed without lookahead bias:
  fractal_in_zone_feature       [0, 1]         price inside active Fibonacci zone
  fractal_zone_dir_feature      [-1, 0, 1]     zone operative direction (bullish/bearish)
  fractal_zone_type_feature     [-1, 0, 1]     LOW / MID / HIGH relative position in cycle
  fractal_fib_position_feature  [-0.5, 1.5]    price position in current cycle Fibonacci
  fractal_deception_feature     [-1, 0, 1]     first_deception signal strength (decays linearly)

Performance: O(n) — one forward pass for zone tracking + one Backtester run for signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ACTIVE_CYCLES = 30   # keep only recent cycles to avoid memory explosion
SIGNAL_DECAY_BARS  = 20  # first_deception signal decays to 0 over this many bars


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def compute_fractal_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute walk-forward fractal features aligned to df.index.

    Args:
        df: OHLCV DataFrame with columns high/low/close.

    Returns:
        DataFrame with 5 *_feature columns, same index as df.
        Returns all-zero features if fractal_strategy is unavailable.
    """
    try:
        from experimental.fractal_strategy.fractals import detect_fractals
        from experimental.fractal_strategy.cycles import (
            compute_fibonacci, compute_zones, MIN_RETRACE_FOR_CYCLE,
        )
        from experimental.fractal_strategy.concurrences import resolve_concurrences
        from experimental.fractal_strategy.models import (
            Cycle, Direction, FractalType, ZoneType,
        )
    except ImportError as exc:
        print(f"⚠️  fractal_strategy unavailable ({exc}) — returning zero fractal features")
        return _zero_features(df)

    df_r = df.reset_index(drop=True)
    n = len(df_r)
    highs  = df_r["high"].values
    lows   = df_r["low"].values
    closes = df_r["close"].values

    # Pre-index fractals by confirmation bar (confirmed_at = central_bar + 2)
    all_fractals = detect_fractals(df_r)
    frac_by_bar: dict[int, list] = {}
    for f in all_fractals:
        frac_by_bar.setdefault(f.confirmed_at, []).append(f)

    _ZONE_TYPE_VAL = {ZoneType.LOW: -1.0, ZoneType.MID: 0.0, ZoneType.HIGH: 1.0}

    # Output arrays
    in_zone_arr   = np.zeros(n, dtype=np.float32)
    zone_dir_arr  = np.zeros(n, dtype=np.float32)
    zone_type_arr = np.zeros(n, dtype=np.float32)
    fib_pos_arr   = np.zeros(n, dtype=np.float32)

    # Walk-forward state
    last_high_frac = None
    last_low_frac  = None
    known_keys: set = set()
    next_id = 0
    active_cycles: list = []
    active_zones:  list = []

    for bar in range(n):
        cc = closes[bar]

        # ── New confirmed fractals → attempt to form cycles ──────────────────
        zones_updated = False
        for frac in frac_by_bar.get(bar, []):
            new_c = None
            if frac.fractal_type == FractalType.HIGH:
                if last_low_frac is not None:
                    new_c = _try_cycle_bullish(
                        last_low_frac, frac, highs, lows, bar,
                        known_keys, next_id,
                        compute_fibonacci, compute_zones,
                        MIN_RETRACE_FOR_CYCLE, Cycle, Direction,
                    )
                    if new_c:
                        known_keys.add((last_low_frac.index, frac.index))
                        next_id += 1
                last_high_frac = frac
            else:  # LOW fractal
                if last_high_frac is not None:
                    new_c = _try_cycle_bearish(
                        last_high_frac, frac, highs, lows, bar,
                        known_keys, next_id,
                        compute_fibonacci, compute_zones,
                        MIN_RETRACE_FOR_CYCLE, Cycle, Direction,
                    )
                    if new_c:
                        known_keys.add((last_high_frac.index, frac.index))
                        next_id += 1
                last_low_frac = frac

            if new_c is not None:
                active_cycles.append(new_c)
                if len(active_cycles) > MAX_ACTIVE_CYCLES:
                    active_cycles = active_cycles[-MAX_ACTIVE_CYCLES:]
                zones_updated = True

        # Recompute valid zones when cycle list changed
        if zones_updated:
            raw_zones = [z for c in active_cycles for z in c.zones]
            active_zones = resolve_concurrences(raw_zones, cc)

        # ── Zone features at this bar ─────────────────────────────────────────
        for zone in active_zones:
            if not zone.contains(cc):
                continue
            in_zone_arr[bar]   = 1.0
            zone_dir_arr[bar]  = 1.0 if zone.direction == Direction.BULLISH else -1.0
            zone_type_arr[bar] = _ZONE_TYPE_VAL[zone.zone_type]

            cycle = next((c for c in active_cycles if c.cycle_id == zone.cycle_id), None)
            if cycle is not None:
                fib_range = abs(cycle.fib.level_100 - cycle.fib.origin)
                if fib_range > 0:
                    pos = (
                        (cc - cycle.fib.origin) / fib_range
                        if cycle.direction == Direction.BULLISH
                        else (cycle.fib.origin - cc) / fib_range
                    )
                    fib_pos_arr[bar] = float(np.clip(pos, -0.5, 1.5))
            break  # use first matching zone only

    # ── First-deception signal feature (separate Backtester pass) ────────────
    decep_arr = _compute_deception_signal(df_r, SIGNAL_DECAY_BARS)

    return pd.DataFrame(
        {
            "fractal_in_zone_feature":      in_zone_arr,
            "fractal_zone_dir_feature":     zone_dir_arr,
            "fractal_zone_type_feature":    zone_type_arr,
            "fractal_fib_position_feature": fib_pos_arr,
            "fractal_deception_feature":    decep_arr,
        },
        index=df.index,
    )


# ──────────────────────────────────────────────────────────────────────────────
# First-deception signal feature
# ──────────────────────────────────────────────────────────────────────────────

def _compute_deception_signal(df_r: pd.DataFrame, decay_bars: int) -> np.ndarray:
    """
    Run the walk-forward backtester (first_deception only) and build a
    signal-strength array that linearly decays to 0 over decay_bars.

    +1.0 → bullish first_deception just fired, 0.0 → no signal / fully decayed.
    """
    try:
        from fractal_strategy import FractalStrategy, Backtester
    except ImportError:
        return np.zeros(len(df_r), dtype=np.float32)

    n = len(df_r)
    arr = np.zeros(n, dtype=np.float32)

    try:
        strategy = FractalStrategy(min_impulse_pips=0.0)
        bt = Backtester(
            strategy,
            allowed_patterns=["first_deception"],
            higher_timeframes=(),   # single-timeframe, no HTF filter
            max_open_bars=200,
        )
        result = bt.run(df_r, walk_forward=True)
    except Exception as exc:
        print(f"⚠️  Backtester failed in fractal feature extraction: {exc}")
        return arr

    from experimental.fractal_strategy.models import Direction

    for trade in result.trades:
        sig = trade.signal
        fired_bar = sig.index
        dir_val   = 1.0 if sig.direction == Direction.BULLISH else -1.0
        # Linear decay from 1.0 at fired_bar to 0 at fired_bar + decay_bars
        end_bar = min(fired_bar + decay_bars, n)
        for b in range(fired_bar, end_bar):
            age      = b - fired_bar
            strength = dir_val * (1.0 - age / decay_bars)
            # Keep the strongest signal if multiple overlap
            if abs(strength) > abs(arr[b]):
                arr[b] = float(strength)

    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Cycle construction helpers (mirrors Backtester._try_cycle_*)
# ──────────────────────────────────────────────────────────────────────────────

def _try_cycle_bullish(
    low_f, high_f, highs, lows, up_to_bar,
    known_keys, cycle_id,
    compute_fibonacci, compute_zones,
    min_retrace, Cycle, Direction,
):
    key = (low_f.index, high_f.index)
    if key in known_keys:
        return None
    impulse = high_f.price - low_f.price
    if impulse <= 0:
        return None
    retrace_start = high_f.confirmed_at
    if retrace_start > up_to_bar:
        return None
    seg = lows[retrace_start: min(retrace_start + 200, up_to_bar + 1)]
    if len(seg) == 0:
        return None
    retrace_low = seg.min()
    if (high_f.price - retrace_low) / impulse < min_retrace:
        return None
    fib = compute_fibonacci(low_f.price, high_f.price, Direction.BULLISH)
    c = Cycle(
        cycle_id=cycle_id, direction=Direction.BULLISH,
        impulse_start=low_f.price, impulse_end=high_f.price,
        retrace_low=retrace_low,
        start_index=low_f.index, end_index=high_f.index,
        retrace_index=retrace_start + int(np.argmin(seg)),
        fib=fib,
    )
    c.zones = compute_zones(c)
    return c


def _try_cycle_bearish(
    high_f, low_f, highs, lows, up_to_bar,
    known_keys, cycle_id,
    compute_fibonacci, compute_zones,
    min_retrace, Cycle, Direction,
):
    key = (high_f.index, low_f.index)
    if key in known_keys:
        return None
    impulse = high_f.price - low_f.price
    if impulse <= 0:
        return None
    retrace_start = low_f.confirmed_at
    if retrace_start > up_to_bar:
        return None
    seg = highs[retrace_start: min(retrace_start + 200, up_to_bar + 1)]
    if len(seg) == 0:
        return None
    retrace_high = seg.max()
    if (retrace_high - low_f.price) / impulse < min_retrace:
        return None
    fib = compute_fibonacci(high_f.price, low_f.price, Direction.BEARISH)
    c = Cycle(
        cycle_id=cycle_id, direction=Direction.BEARISH,
        impulse_start=high_f.price, impulse_end=low_f.price,
        retrace_low=retrace_high,
        start_index=high_f.index, end_index=low_f.index,
        retrace_index=retrace_start + int(np.argmax(seg)),
        fib=fib,
    )
    c.zones = compute_zones(c)
    return c


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _zero_features(df: pd.DataFrame) -> pd.DataFrame:
    z = np.zeros(len(df), dtype=np.float32)
    return pd.DataFrame(
        {
            "fractal_in_zone_feature":      z.copy(),
            "fractal_zone_dir_feature":     z.copy(),
            "fractal_zone_type_feature":    z.copy(),
            "fractal_fib_position_feature": z.copy(),
            "fractal_deception_feature":    z.copy(),
        },
        index=df.index,
    )
