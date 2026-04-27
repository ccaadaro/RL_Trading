"""
utils/implementation_shortfall.py

Tracks Implementation Shortfall (IS) per trade.

IS = (Average Fill Price - Decision Price) / Decision Price  [for buys]
   = (Decision Price - Average Fill Price) / Decision Price  [for sells]

Positive IS = we paid more (or sold for less) than intended → cost.
Negative IS = favorable slippage (rare but possible with passive execution).
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ExecutionRecord:
    """Immutable record of a single order execution schedule."""
    trade_id:       str
    side:           str          # "buy" or "sell"
    decision_price: float        # Mid-price at signal time
    decision_time:  float        # Unix timestamp

    # Filled progressively via add_fill()
    total_target:   float   = 0.0   # Total notional intended
    fills:          list    = field(default_factory=list)  # [(price, qty, ts), ...]

    @property
    def filled_qty(self) -> float:
        return sum(q for _, q, _ in self.fills)

    @property
    def fill_pct(self) -> float:
        return self.filled_qty / self.total_target if self.total_target > 0 else 0.0

    @property
    def avg_fill_price(self) -> Optional[float]:
        if not self.fills:
            return None
        total_notional = sum(p * q for p, q, _ in self.fills)
        total_qty      = sum(q for _, q, _ in self.fills)
        return total_notional / total_qty if total_qty > 0 else None

    @property
    def implementation_shortfall(self) -> Optional[float]:
        """IS in basis points (bps). Positive = cost."""
        afp = self.avg_fill_price
        if afp is None or self.decision_price == 0:
            return None
        if self.side == "buy":
            return (afp - self.decision_price) / self.decision_price * 10_000
        else:
            return (self.decision_price - afp) / self.decision_price * 10_000

    @property
    def execution_duration_s(self) -> Optional[float]:
        if not self.fills:
            return None
        return self.fills[-1][2] - self.decision_time

    def add_fill(self, price: float, qty: float, ts: Optional[float] = None):
        self.fills.append((price, qty, ts or time.time()))

    def to_dict(self) -> dict:
        return {
            "trade_id":        self.trade_id,
            "side":            self.side,
            "decision_price":  self.decision_price,
            "avg_fill_price":  self.avg_fill_price,
            "IS_bps":          self.implementation_shortfall,
            "fill_pct":        self.fill_pct,
            "duration_s":      self.execution_duration_s,
        }


class ISTracker:
    """
    Maintains a rolling log of all ExecutionRecords.
    Provides aggregate statistics for auditing.
    """
    def __init__(self):
        self._records: dict[str, ExecutionRecord] = {}
        self._closed:  list[ExecutionRecord]       = []

    def open_trade(self, trade_id: str, side: str, decision_price: float,
                   total_target: float, decision_time: Optional[float] = None) -> ExecutionRecord:
        rec = ExecutionRecord(
            trade_id=trade_id,
            side=side,
            decision_price=decision_price,
            decision_time=decision_time or time.time(),
            total_target=total_target,
        )
        self._records[trade_id] = rec
        return rec

    def add_fill(self, trade_id: str, price: float, qty: float, ts: Optional[float] = None):
        if trade_id in self._records:
            self._records[trade_id].add_fill(price, qty, ts)

    def close_trade(self, trade_id: str) -> Optional[ExecutionRecord]:
        rec = self._records.pop(trade_id, None)
        if rec:
            self._closed.append(rec)
        return rec

    def summary(self) -> dict:
        if not self._closed:
            return {"n_trades": 0}
        is_vals = [r.implementation_shortfall for r in self._closed if r.implementation_shortfall is not None]
        return {
            "n_trades":      len(self._closed),
            "mean_IS_bps":   np.mean(is_vals),
            "median_IS_bps": np.median(is_vals),
            "p95_IS_bps":    np.percentile(is_vals, 95),
            "pct_positive":  np.mean([v > 0 for v in is_vals]) * 100,
        }

    def print_summary(self):
        s = self.summary()
        print("\n=== Implementation Shortfall Report ===")
        for k, v in s.items():
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
