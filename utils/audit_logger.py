"""
utils/audit_logger.py
──────────────────────
MiFID II / Dodd-Frank compliant trade and risk audit logger.

Writes structured JSONL records (one JSON object per line) so logs are:
  - Append-only, never overwritten
  - Easily parseable with pandas: pd.read_json(..., lines=True)
  - Machine-readable for regulatory submission

Record types
────────────
  TRADE        — every position open/close with price, size, P&L
  RISK_EVENT   — every time a risk override fires (DD, turbulence, stop)
  DAILY_SUMMARY— end-of-day performance snapshot

Usage
─────
    from utils.audit_logger import AuditLogger

    audit = AuditLogger(log_dir="logs/audit", pair="BTC/USDT")

    # On each bar
    audit.log_trade(
        timestamp=bar_timestamp,
        action="LONG",          # "LONG" | "HOLD" | "FLAT"
        price=close_price,
        portfolio_value=nav,
        drawdown_pct=dd,
        turbulence=turb,
        top_features={"rsi": 0.03, "turbulence": 0.02},  # from explainability
    )

    # When a risk override fires
    audit.log_risk_event(
        timestamp=bar_timestamp,
        trigger="turbulence",   # "turbulence" | "drawdown" | "trailing_stop" | "cooldown"
        turbulence=turb,
        drawdown_pct=dd,
        action_intended=1.0,    # what the model wanted
        action_taken=0.0,       # what risk override enforced
    )

    # At end of day
    audit.write_daily_summary(date="2026-04-08")
"""

import json
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# AuditLogger
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe JSONL audit logger for RL trading systems.

    All writes are protected by a lock so multiple strategies / processes
    can safely append to the same log file.

    Parameters
    ----------
    log_dir : directory where JSONL files are written.
    pair : trading pair label appended to every record (e.g. "BTC/USDT").
    strategy_id : identifies the strategy version / model run.
    """

    def __init__(
        self,
        log_dir:     str = "logs/audit",
        pair:        str = "BTC/USDT",
        strategy_id: str = "RLEnsemble",
    ):
        self.log_dir     = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pair        = pair
        self.strategy_id = strategy_id
        self._lock       = threading.Lock()

        # In-memory buffer for daily summary calculation
        self._today_trades:    list = []
        self._today_nav:       list = []
        self._today_risk_events: int = 0

    # ── internal write ────────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        """Append one JSON record to today's log file."""
        today     = datetime.date.today().isoformat()
        log_file  = self.log_dir / f"audit_{today}.jsonl"
        line      = json.dumps(record, default=str)          # str() handles datetimes
        with self._lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ── public API ────────────────────────────────────────────────────────────

    def log_trade(
        self,
        timestamp:       Any,
        action:          str,           # "LONG" | "HOLD" | "FLAT"
        price:           float,
        portfolio_value: float,
        drawdown_pct:    float = 0.0,
        turbulence:      float = 0.0,
        top_features:    Optional[Dict[str, float]] = None,
        gen_ratio:       float = float("nan"),
        seed:            int = -1,
        extra:           Optional[Dict] = None,
    ) -> None:
        """
        Log a single bar's trading decision.

        Required for MiFID II Article 25 (record-keeping of transactions) and
        Dodd-Frank Section 729 (swap reporting).  The `top_features` dict
        provides the explainability required under MiFID II Article 24.
        """
        record: Dict[str, Any] = {
            "record_type":     "TRADE",
            "logged_at":       datetime.datetime.utcnow().isoformat() + "Z",
            "bar_timestamp":   str(timestamp),
            "pair":            self.pair,
            "strategy_id":     self.strategy_id,
            "action":          action,
            "price":           round(float(price), 6),
            "portfolio_value": round(float(portfolio_value), 4),
            "drawdown_pct":    round(float(drawdown_pct), 6),
            "turbulence":      round(float(turbulence), 6),
            "gen_ratio":       float(gen_ratio),
            "seed":            seed,
            "top_features":    top_features or {},
        }
        if extra:
            record.update(extra)

        self._write(record)
        self._today_trades.append(action)
        self._today_nav.append(float(portfolio_value))

    def log_risk_event(
        self,
        timestamp:       Any,
        trigger:         str,           # "turbulence"|"drawdown"|"trailing_stop"|"cooldown"
        action_intended: float,         # what the model voted (0.0 or 1.0)
        action_taken:    float,         # what was actually executed after override
        turbulence:      float = 0.0,
        drawdown_pct:    float = 0.0,
        cooldown_bars:   int   = 0,
        extra:           Optional[Dict] = None,
    ) -> None:
        """
        Log a risk-management override event.

        Required under MiFID II RTS 6 (algorithmic trading controls) —
        every automated pre-trade risk check result must be recorded.
        """
        record: Dict[str, Any] = {
            "record_type":     "RISK_EVENT",
            "logged_at":       datetime.datetime.utcnow().isoformat() + "Z",
            "bar_timestamp":   str(timestamp),
            "pair":            self.pair,
            "strategy_id":     self.strategy_id,
            "trigger":         trigger,
            "action_intended": float(action_intended),
            "action_taken":    float(action_taken),
            "turbulence":      round(float(turbulence), 6),
            "drawdown_pct":    round(float(drawdown_pct), 6),
            "cooldown_bars":   cooldown_bars,
        }
        if extra:
            record.update(extra)

        self._write(record)
        self._today_risk_events += 1

    def write_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Write an end-of-day summary record and return it.

        Includes: total trades, % time in market, daily return, max intra-day DD,
        number of risk overrides.  Resets daily buffers after writing.
        """
        import numpy as np

        date = date or datetime.date.today().isoformat()
        nav  = np.array(self._today_nav, dtype=float)

        daily_return   = float(nav[-1] / nav[0] - 1) * 100 if len(nav) > 1 else 0.0
        peaks          = np.maximum.accumulate(nav) if len(nav) > 0 else np.array([1.0])
        max_dd         = float(np.max(1 - nav / peaks)) * 100 if len(nav) > 1 else 0.0
        pct_long       = (
            self._today_trades.count("LONG") / max(len(self._today_trades), 1) * 100
        )
        n_changes      = sum(
            1 for i in range(1, len(self._today_trades))
            if self._today_trades[i] != self._today_trades[i - 1]
        )

        summary: Dict[str, Any] = {
            "record_type":    "DAILY_SUMMARY",
            "logged_at":      datetime.datetime.utcnow().isoformat() + "Z",
            "date":           date,
            "pair":           self.pair,
            "strategy_id":    self.strategy_id,
            "n_bars":         len(self._today_trades),
            "n_position_changes": n_changes,
            "pct_long":       round(pct_long, 2),
            "daily_return_pct": round(daily_return, 4),
            "max_intraday_dd_pct": round(max_dd, 4),
            "start_nav":      round(float(nav[0]),  4) if len(nav) else 0.0,
            "end_nav":        round(float(nav[-1]), 4) if len(nav) else 0.0,
            "n_risk_overrides": self._today_risk_events,
        }

        self._write(summary)

        # Reset buffers
        self._today_trades    = []
        self._today_nav       = []
        self._today_risk_events = 0

        return summary

    # ── report reading ────────────────────────────────────────────────────────

    @classmethod
    def load_logs(
        cls,
        log_dir: str = "logs/audit",
        start:   Optional[str] = None,
        end:     Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> "pd.DataFrame":
        """
        Load all JSONL audit records into a DataFrame.

        Parameters
        ----------
        log_dir : directory containing audit_*.jsonl files.
        start, end : ISO date strings for filtering (e.g. "2026-01-01").
        record_type : filter to "TRADE", "RISK_EVENT", or "DAILY_SUMMARY".

        Returns
        -------
        pd.DataFrame with one row per record, sorted by bar_timestamp.
        """
        import pandas as pd

        path    = Path(log_dir)
        records = []
        for f in sorted(path.glob("audit_*.jsonl")):
            # Filter by date from filename (audit_YYYY-MM-DD.jsonl)
            stem = f.stem.replace("audit_", "")
            if start and stem < start:
                continue
            if end   and stem > end:
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if record_type:
            df = df[df["record_type"] == record_type]
        if "bar_timestamp" in df.columns:
            df = df.sort_values("bar_timestamp").reset_index(drop=True)
        return df
