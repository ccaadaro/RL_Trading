"""
Backtester para la estrategia Fractal-Fibonacci.

Diseño sin lookahead:
  - Las señales se generan en la barra i (fractales confirmados con 2 velas de retraso).
  - La orden se coloca al cierre de la barra i como orden límite.
  - Se simula barra a barra hacia adelante para detectar fill, stop o target.
  - El tamaño de posición usa riesgo fijo (% del equity en cada operación).

Uso:
    from fractal_strategy import FractalStrategy, Backtester

    strategy = FractalStrategy(min_impulse_pips=0.001)
    bt = Backtester(strategy, initial_capital=10_000, risk_pct=0.01)
    result = bt.run(df)

    print(result.metrics)
    print(result.trades_df)
    result.equity_curve.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .concurrences import resolve_concurrences
from .cycles import MIN_RETRACE_FOR_CYCLE, compute_fibonacci, compute_zones
from .fractals import detect_fractals
from .models import (
    Cycle, Direction, FractalType, Signal, SignalType, Zone,
)
from .patterns import (
    detect_deep_entry, detect_double_bottom_top,
    detect_extreme_deception, detect_first_deception,
)
from .strategy import FractalStrategy


class TradeOutcome(Enum):
    WIN_TARGET = "win_target"
    WIN_PARTIAL = "win_partial"    # Cerró 50% al llegar a 61.8 de seguimiento
    LOSS_STOP = "loss_stop"
    OPEN = "open"
    CANCELLED = "cancelled"       # Orden límite nunca se ejecutó


@dataclass
class Trade:
    signal: Signal
    entry_bar: int
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    position_size: float           # Unidades del activo
    risk_amount: float             # Capital en riesgo

    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    outcome: TradeOutcome = TradeOutcome.OPEN

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        direction_mult = 1.0 if self.signal.direction == Direction.BULLISH else -1.0
        return direction_mult * (self.exit_price - self.entry_price) * self.position_size

    @property
    def pnl_r(self) -> float:
        """PnL expresado en múltiplos de R (riesgo)."""
        if self.risk_amount == 0:
            return 0.0
        return self.pnl / self.risk_amount

    def to_dict(self) -> dict:
        return {
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "direction": self.signal.direction.value,
            "pattern": self.signal.pattern.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "position_size": self.position_size,
            "risk_amount": self.risk_amount,
            "pnl": self.pnl,
            "pnl_r": self.pnl_r,
            "outcome": self.outcome.value,
        }


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    initial_capital: float

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.outcome not in (TradeOutcome.OPEN, TradeOutcome.CANCELLED)]

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    @property
    def metrics(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"error": "Sin operaciones cerradas"}

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        pnl_r_series = [t.pnl_r for t in closed]
        equity = self.equity_curve

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Sharpe anualizado (asumiendo barras diarias; ajustar factor si es H1/H4)
        returns = equity.pct_change().dropna()
        sharpe = (
            float(returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0 else 0.0
        )

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed),
            "avg_win_r": float(np.mean([t.pnl_r for t in wins])) if wins else 0.0,
            "avg_loss_r": float(np.mean([t.pnl_r for t in losses])) if losses else 0.0,
            "profit_factor": profit_factor,
            "expectancy_r": float(np.mean(pnl_r_series)),
            "total_pnl": float(sum(t.pnl for t in closed)),
            "final_equity": float(equity.iloc[-1]),
            "return_pct": float((equity.iloc[-1] / self.initial_capital - 1) * 100),
            "max_drawdown_pct": float(max_drawdown * 100),
            "sharpe_ratio": sharpe,
        }


class Backtester:
    """
    Simula la ejecución de la estrategia sobre datos históricos.

    Args:
        strategy: Instancia configurada de FractalStrategy.
        initial_capital: Capital inicial en la divisa de la cuenta.
        risk_pct: Fracción del equity a arriesgar por operación (0.01 = 1%).
        max_fill_bars: Máximo de barras para esperar que se ejecute una orden límite.
        max_open_bars: Máximo de barras que puede estar abierta una operación.
        partial_close: Si True, cierra el 50% de la posición al llegar al 61.8%
                       del fibo de seguimiento (regla de la estrategia).
    """

    def __init__(
        self,
        strategy: FractalStrategy,
        initial_capital: float = 10_000.0,
        risk_pct: float = 0.01,
        max_fill_bars: int = 20,
        max_open_bars: int = 200,
        partial_close: bool = True,
        max_active_cycles: int = 30,
        allowed_patterns: list[str] | None = None,
        higher_timeframes: tuple[int, ...] = (4, 24),
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.max_fill_bars = max_fill_bars
        self.max_open_bars = max_open_bars
        self.partial_close = partial_close
        self.max_active_cycles = max_active_cycles
        # None = todos los patrones; lista de strings = solo esos
        # Ej: ["first_deception", "second_deception"]
        self.allowed_patterns = set(allowed_patterns) if allowed_patterns else None
        # Multiplicadores de temporalidad superior respecto a la base (ej. 4 = 4H si base es 1H)
        # Los ciclos y zonas de temporalidades superiores se combinan con los de la base
        self.higher_timeframes = tuple(higher_timeframes) if higher_timeframes else ()

    def run(self, df: pd.DataFrame, walk_forward: bool = True) -> BacktestResult:
        """
        Ejecuta el backtest completo.

        Args:
            df: DataFrame OHLCV con índice numérico o datetime.
            walk_forward: Si True (por defecto), construye ciclos y señales
                incrementalmente sin lookahead. Si False, usa todos los datos
                de golpe (más rápido pero con sesgo de información futura).

        Returns:
            BacktestResult con trades, equity curve y métricas.
        """
        df = df.reset_index(drop=True)
        if walk_forward:
            return self._run_walk_forward(df)
        result = self.strategy.run(df)
        signals: list[Signal] = result["signals"]

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        trades: list[Trade] = []
        equity = self.initial_capital
        equity_history = [equity] * (len(df) + 1)

        # Índice de señales para búsqueda rápida
        signal_by_bar: dict[int, list[Signal]] = {}
        for sig in signals:
            signal_by_bar.setdefault(sig.index, []).append(sig)

        open_trades: list[Trade] = []
        pending_orders: list[tuple[int, Signal]] = []  # (placed_at_bar, signal)

        for bar in range(n):
            current_high = highs[bar]
            current_low = lows[bar]
            current_close = closes[bar]

            # --- 1. Actualizar operaciones abiertas ---
            still_open = []
            for trade in open_trades:
                closed = self._update_trade(
                    trade, bar, current_high, current_low, current_close
                )
                if closed:
                    equity += trade.pnl
                else:
                    # Forzar cierre si lleva demasiado tiempo abierta
                    if bar - trade.entry_bar >= self.max_open_bars:
                        trade.exit_bar = bar
                        trade.exit_price = current_close
                        trade.outcome = TradeOutcome.WIN_TARGET if trade.pnl > 0 else TradeOutcome.LOSS_STOP
                        equity += trade.pnl
                    else:
                        still_open.append(trade)
            open_trades = still_open

            # --- 2. Intentar ejecutar órdenes pendientes ---
            still_pending = []
            for placed_bar, sig in pending_orders:
                if bar - placed_bar > self.max_fill_bars:
                    # Caducada
                    t = self._build_trade(sig, bar, equity)
                    t.outcome = TradeOutcome.CANCELLED
                    trades.append(t)
                    continue

                filled = self._try_fill(sig, bar, current_high, current_low, equity)
                if filled is not None:
                    open_trades.append(filled)
                    trades.append(filled)
                else:
                    still_pending.append((placed_bar, sig))
            pending_orders = still_pending

            # --- 3. Nuevas señales en esta barra ---
            for sig in signal_by_bar.get(bar, []):
                pending_orders.append((bar, sig))

            equity_history[bar + 1] = equity

        # Cerrar operaciones abiertas al final con precio de cierre
        for trade in open_trades:
            trade.exit_bar = n - 1
            trade.exit_price = closes[-1]
            trade.outcome = TradeOutcome.WIN_TARGET if trade.pnl > 0 else TradeOutcome.LOSS_STOP
            equity += trade.pnl

        equity_curve = pd.Series(equity_history[1:], index=df.index)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
        )

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Reagrupa el DataFrame de barras base en velas de n barras (OHLCV)."""
        idx = df.index // n
        return df.groupby(idx).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).reset_index(drop=True)

    def _run_walk_forward(self, df: pd.DataFrame) -> BacktestResult:
        """
        Walk-forward estricto O(n): en la barra i solo usa datos hasta i.

        Optimización clave: cuando llega un nuevo fractal, solo comprueba
        el par inmediato anterior (LOW→HIGH o HIGH→LOW), nunca reescanea
        todos los históricos. Esto convierte el algoritmo de O(n²) a O(n).

        Multi-temporalidad: los ciclos y zonas de temporalidades superiores
        (4H, Daily, etc.) se combinan con los de la base para la detección
        de patrones, replicando el análisis top-down de la estrategia.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        all_fractals = detect_fractals(df)
        frac_by_confirm: dict[int, list] = {}
        for f in all_fractals:
            frac_by_confirm.setdefault(f.confirmed_at, []).append(f)

        # ── Pre-computar datos y fractales de temporalidades superiores ──────
        # Las zonas HTF actúan como filtro de confirmación: solo se generan
        # señales sobre zonas 1H que están dentro de una zona HTF del mismo
        # sentido. Los ciclos HTF NO se mezclan con los 1H en resolve_concurrences.
        htf_highs: dict[int, np.ndarray] = {}
        htf_lows: dict[int, np.ndarray] = {}
        htf_frac_by_confirm: dict[int, dict[int, list]] = {}
        htf_state: dict[int, dict] = {}

        for mult in self.higher_timeframes:
            htf_df = self._resample_ohlcv(df, mult)
            htf_highs[mult] = htf_df["high"].values
            htf_lows[mult] = htf_df["low"].values
            htf_fracs = detect_fractals(htf_df)
            htf_frac_by_confirm[mult] = {}
            for f in htf_fracs:
                htf_frac_by_confirm[mult].setdefault(f.confirmed_at, []).append(f)
            htf_state[mult] = {
                "last_high_frac": None,
                "last_low_frac": None,
                "known_keys": set(),
                "next_id": 100_000 + mult * 10_000,
                "active_cycles": [],
                "prev_htf_bar": -1,
                "zones": [],          # zonas válidas de este TF (resueltas entre sí)
            }

        # Zonas HTF combinadas de todos los multiplicadores (para filtro rápido)
        htf_zones: list[Zone] = []

        # Último fractal confirmado de cada tipo (para pareo O(1))
        last_high_frac = None
        last_low_frac = None

        known_cycle_keys: set = set()
        next_cycle_id = 0
        active_cycles: list[Cycle] = []
        active_zones: list[Zone] = []
        generated_signal_keys: set = set()

        trades: list[Trade] = []
        open_trades: list[Trade] = []
        pending_orders: list[tuple[int, Signal]] = []
        equity = self.initial_capital
        equity_history = [equity] * (n + 1)
        signal_by_bar: dict[int, list[Signal]] = {}

        for bar in range(n):
            ch = highs[bar]
            cl = lows[bar]
            cc = closes[bar]

            zones_updated = False

            # ── 1. Nuevos fractales confirmados en esta barra (base TF) ───────
            new_fracs = frac_by_confirm.get(bar, [])
            new_cycles: list[Cycle] = []

            for frac in new_fracs:
                if frac.fractal_type == FractalType.HIGH:
                    # Ciclo alcista: LOW anterior → este HIGH
                    if last_low_frac is not None:
                        c = self._try_cycle_bullish(
                            last_low_frac, frac, highs, lows, bar,
                            known_cycle_keys, next_cycle_id,
                            self.strategy.min_impulse_pips,
                        )
                        if c:
                            new_cycles.append(c)
                            known_cycle_keys.add((last_low_frac.index, frac.index))
                            next_cycle_id += 1
                    last_high_frac = frac

                else:  # LOW
                    # Ciclo bajista: HIGH anterior → este LOW
                    if last_high_frac is not None:
                        c = self._try_cycle_bearish(
                            last_high_frac, frac, highs, lows, bar,
                            known_cycle_keys, next_cycle_id,
                            self.strategy.min_impulse_pips,
                        )
                        if c:
                            new_cycles.append(c)
                            known_cycle_keys.add((last_high_frac.index, frac.index))
                            next_cycle_id += 1
                    last_low_frac = frac

            if new_cycles:
                active_cycles.extend(new_cycles)
                if len(active_cycles) > self.max_active_cycles:
                    active_cycles = active_cycles[-self.max_active_cycles:]
                zones_updated = True

            # ── 2. Actualizar ciclos de temporalidades superiores ──────────────
            htf_zones_updated = False
            for mult in self.higher_timeframes:
                htf_bar = (bar + 1) // mult - 1  # última barra HTF completa
                state = htf_state[mult]
                if htf_bar > state["prev_htf_bar"] and htf_bar >= 0:
                    new_htf_cycles = False
                    for htf_idx in range(state["prev_htf_bar"] + 1, htf_bar + 1):
                        for frac in htf_frac_by_confirm[mult].get(htf_idx, []):
                            c = None
                            if frac.fractal_type == FractalType.HIGH:
                                if state["last_low_frac"] is not None:
                                    c = self._try_cycle_bullish(
                                        state["last_low_frac"], frac,
                                        htf_highs[mult], htf_lows[mult], htf_bar,
                                        state["known_keys"], state["next_id"],
                                        self.strategy.min_impulse_pips,
                                    )
                                    if c:
                                        state["known_keys"].add((state["last_low_frac"].index, frac.index))
                                        state["next_id"] += 1
                                state["last_high_frac"] = frac
                            else:
                                if state["last_high_frac"] is not None:
                                    c = self._try_cycle_bearish(
                                        state["last_high_frac"], frac,
                                        htf_highs[mult], htf_lows[mult], htf_bar,
                                        state["known_keys"], state["next_id"],
                                        self.strategy.min_impulse_pips,
                                    )
                                    if c:
                                        state["known_keys"].add((state["last_high_frac"].index, frac.index))
                                        state["next_id"] += 1
                                state["last_low_frac"] = frac
                            if c:
                                state["active_cycles"].append(c)
                                new_htf_cycles = True
                    if len(state["active_cycles"]) > self.max_active_cycles:
                        state["active_cycles"] = state["active_cycles"][-self.max_active_cycles:]
                    state["prev_htf_bar"] = htf_bar
                    if new_htf_cycles:
                        # Recalcular zonas de este TF por separado
                        state["zones"] = resolve_concurrences(
                            [z for c in state["active_cycles"] for z in c.zones], cc
                        )
                        htf_zones_updated = True

            if htf_zones_updated:
                htf_zones = [
                    z for mult in self.higher_timeframes
                    for z in htf_state[mult]["zones"]
                ]

            # ── 3. Recalcular zonas 1H cuando hay ciclos nuevos ───────────────
            if zones_updated:
                active_zones = resolve_concurrences(
                    [z for c in active_cycles for z in c.zones], cc
                )

            # ── 5. Buscar señales en cada barra donde el precio toca una zona ──
            for zone in active_zones:
                if not zone.contains(cc):
                    continue

                # Filtro HTF: si hay temporalidades superiores configuradas,
                # solo operar en zonas 1H que se solapen con una zona HTF
                # del mismo sentido operativo (confirmación top-down).
                if self.higher_timeframes:
                    htf_confirmed = any(
                        hz.direction == zone.direction
                        and hz.lower < zone.upper and zone.lower < hz.upper
                        for hz in htf_zones
                    )
                    if not htf_confirmed:
                        continue

                cycle = next(
                    (c for c in active_cycles if c.cycle_id == zone.cycle_id), None
                )
                if cycle is None:
                    continue
                start = max(0, cycle.retrace_index)
                next_lvl = self._get_next_level_wf(zone, active_zones, cycle)
                df_so_far = df.iloc[: bar + 1].reset_index(drop=True)
                for detector in [
                    lambda z=zone, c=cycle, s=start: detect_first_deception(df_so_far, z, c, s),
                    lambda z=zone, c=cycle, s=start: detect_deep_entry(df_so_far, z, c, s),
                    lambda z=zone, c=cycle, s=start, nl=next_lvl: detect_extreme_deception(df_so_far, z, nl, c, s),
                    lambda z=zone, c=cycle, s=start: detect_double_bottom_top(df_so_far, z, c, s),
                ]:
                    sig = detector()
                    if sig is None:
                        continue
                    key = (sig.index, sig.pattern.value, sig.direction.value)
                    if key in generated_signal_keys:
                        break
                    if self.allowed_patterns and sig.pattern.value not in self.allowed_patterns:
                        break
                    generated_signal_keys.add(key)
                    signal_by_bar.setdefault(sig.index, []).append(sig)
                    break

            # ── 3. Gestión de operaciones ──────────────────────────────────────
            still_open = []
            for trade in open_trades:
                closed = self._update_trade(trade, bar, ch, cl, cc)
                if closed:
                    equity += trade.pnl
                elif bar - trade.entry_bar >= self.max_open_bars:
                    trade.exit_bar = bar
                    trade.exit_price = cc
                    trade.outcome = (
                        TradeOutcome.WIN_TARGET if trade.pnl > 0 else TradeOutcome.LOSS_STOP
                    )
                    equity += trade.pnl
                else:
                    still_open.append(trade)
            open_trades = still_open

            still_pending = []
            for placed_bar, sig in pending_orders:
                if bar - placed_bar > self.max_fill_bars:
                    t = self._build_trade(sig, bar, equity)
                    t.outcome = TradeOutcome.CANCELLED
                    trades.append(t)
                    continue
                filled = self._try_fill(sig, bar, ch, cl, equity)
                if filled is not None:
                    open_trades.append(filled)
                    trades.append(filled)
                else:
                    still_pending.append((placed_bar, sig))
            pending_orders = still_pending

            for sig in signal_by_bar.get(bar, []):
                pending_orders.append((bar, sig))

            equity_history[bar + 1] = equity

        for trade in open_trades:
            trade.exit_bar = n - 1
            trade.exit_price = closes[-1]
            trade.outcome = TradeOutcome.WIN_TARGET if trade.pnl > 0 else TradeOutcome.LOSS_STOP
            equity += trade.pnl

        return BacktestResult(
            trades=trades,
            equity_curve=pd.Series(equity_history[1:], index=df.index),
            initial_capital=self.initial_capital,
        )

    def _try_cycle_bullish(self, low_f, high_f, highs, lows, up_to_bar,
                            known_keys, cycle_id, min_impulse_pips):
        key = (low_f.index, high_f.index)
        if key in known_keys:
            return None
        impulse_size = high_f.price - low_f.price
        if impulse_size < min_impulse_pips or impulse_size <= 0:
            return None
        retrace_start = high_f.confirmed_at
        if retrace_start > up_to_bar:
            return None
        retrace_end = min(retrace_start + 200, up_to_bar + 1)
        segment = lows[retrace_start:retrace_end]
        if len(segment) == 0:
            return None
        retrace_low = segment.min()
        if (high_f.price - retrace_low) / impulse_size < MIN_RETRACE_FOR_CYCLE:
            return None
        retrace_idx = retrace_start + int(np.argmin(segment))
        fib = compute_fibonacci(low_f.price, high_f.price, Direction.BULLISH)
        c = Cycle(
            cycle_id=cycle_id, direction=Direction.BULLISH,
            impulse_start=low_f.price, impulse_end=high_f.price,
            retrace_low=retrace_low, start_index=low_f.index,
            end_index=high_f.index, retrace_index=retrace_idx, fib=fib,
        )
        c.zones = compute_zones(c)
        return c

    def _try_cycle_bearish(self, high_f, low_f, highs, lows, up_to_bar,
                            known_keys, cycle_id, min_impulse_pips):
        key = (high_f.index, low_f.index)
        if key in known_keys:
            return None
        impulse_size = high_f.price - low_f.price
        if impulse_size < min_impulse_pips or impulse_size <= 0:
            return None
        retrace_start = low_f.confirmed_at
        if retrace_start > up_to_bar:
            return None
        retrace_end = min(retrace_start + 200, up_to_bar + 1)
        segment = highs[retrace_start:retrace_end]
        if len(segment) == 0:
            return None
        retrace_high = segment.max()
        if (retrace_high - low_f.price) / impulse_size < MIN_RETRACE_FOR_CYCLE:
            return None
        retrace_idx = retrace_start + int(np.argmax(segment))
        fib = compute_fibonacci(high_f.price, low_f.price, Direction.BEARISH)
        c = Cycle(
            cycle_id=cycle_id, direction=Direction.BEARISH,
            impulse_start=high_f.price, impulse_end=low_f.price,
            retrace_low=retrace_high, start_index=high_f.index,
            end_index=low_f.index, retrace_index=retrace_idx, fib=fib,
        )
        c.zones = compute_zones(c)
        return c

    def _get_next_level_wf(
        self, zone: Zone, all_zones: list[Zone], cycle: Cycle
    ) -> float:
        from .models import Direction
        fib = cycle.fib
        if zone.direction == Direction.BULLISH:
            candidates = [
                z.lower for z in all_zones
                if z.lower > zone.upper and z.direction == Direction.BEARISH
            ]
            candidates.append(fib.ext_1382)
            return min(candidates)
        else:
            candidates = [
                z.upper for z in all_zones
                if z.upper < zone.lower and z.direction == Direction.BULLISH
            ]
            candidates.append(fib.ext_1382)
            return max(candidates)

    def _try_fill(
        self,
        sig: Signal,
        bar: int,
        high: float,
        low: float,
        equity: float,
    ) -> Optional[Trade]:
        """Intenta ejecutar una orden límite. Devuelve Trade si se ejecuta."""
        if sig.signal_type == SignalType.ENTER_AGGRESSIVE:
            # Orden de mercado: se ejecuta en el open de la siguiente barra
            # (simulamos con el precio de la señal)
            filled = True
        else:
            # Orden límite: se ejecuta cuando el precio toca entry_price
            if sig.direction == Direction.BULLISH:
                filled = low <= sig.entry_price
            else:
                filled = high >= sig.entry_price

        if not filled:
            return None

        return self._build_trade(sig, bar, equity)

    def _build_trade(self, sig: Signal, bar: int, equity: float) -> Trade:
        """Construye un Trade con tamaño de posición basado en riesgo fijo."""
        risk_per_unit = abs(sig.entry_price - sig.stop_price)
        risk_amount = equity * self.risk_pct
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0

        return Trade(
            signal=sig,
            entry_bar=bar,
            entry_price=sig.entry_price,
            stop_price=sig.stop_price,
            target_price=sig.target_price,
            position_size=position_size,
            risk_amount=risk_amount,
        )

    def _update_trade(
        self,
        trade: Trade,
        bar: int,
        high: float,
        low: float,
        close: float,
    ) -> bool:
        """
        Actualiza el estado de una operación abierta.
        Devuelve True si la operación fue cerrada.
        """
        direction = trade.signal.direction

        # Comprobar stop
        stop_hit = (
            low <= trade.stop_price if direction == Direction.BULLISH
            else high >= trade.stop_price
        )
        if stop_hit:
            trade.exit_bar = bar
            trade.exit_price = trade.stop_price
            trade.outcome = TradeOutcome.LOSS_STOP
            return True

        # Comprobar target
        if trade.target_price is not None:
            target_hit = (
                high >= trade.target_price if direction == Direction.BULLISH
                else low <= trade.target_price
            )
            if target_hit:
                trade.exit_bar = bar
                trade.exit_price = trade.target_price
                trade.outcome = TradeOutcome.WIN_TARGET
                return True

        return False
