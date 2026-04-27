"""
fractal_strategy — Implementación de la estrategia Fractal-Fibonacci (Joaquín Vega).

API pública:
    FractalStrategy     — Orquestador principal
    Backtester          — Simulación histórica sin lookahead
    Direction           — Enum: BULLISH / BEARISH
    SignalType          — Enum: ENTER_AGGRESSIVE / ENTER_CALM / CANCEL / CLOSE_HALF
    PatternType         — Enum: tipos de patrón de entrada
    Signal              — Dataclass con la señal de trading
    Cycle               — Dataclass con el ciclo operable
    Zone                — Dataclass con la zona de trabajo
    Fractal             — Dataclass con el fractal detectado
    detect_fractals     — Función standalone de detección de fractales
    identify_cycles     — Función standalone de identificación de ciclos
    resolve_concurrences — Función standalone de resolución de concurrencias

Ejemplo de uso:
    import pandas as pd
    from fractal_strategy import FractalStrategy, Backtester

    df = pd.read_csv("eurusd_h4.csv")
    strategy = FractalStrategy(min_impulse_pips=0.001)

    # Análisis
    result = strategy.run(df)
    for signal in result["signals"]:
        print(signal.to_dict())

    # Backtest
    bt = Backtester(strategy, initial_capital=10_000, risk_pct=0.01)
    bt_result = bt.run(df)
    print(bt_result.metrics)
    print(bt_result.trades_df)
"""

from .backtester import Backtester, BacktestResult, Trade, TradeOutcome
from .cycles import identify_cycles
from .concurrences import resolve_concurrences
from .fractals import detect_fractals, find_control_points
from .models import (
    Cycle,
    ConcurrenceType,
    Direction,
    DeceptionZone,
    Fractal,
    FibonacciLevels,
    FractalType,
    PatternType,
    Signal,
    SignalType,
    Zone,
    ZoneType,
)
from .strategy import FractalStrategy

__all__ = [
    "FractalStrategy",
    "Backtester",
    "BacktestResult",
    "Trade",
    "TradeOutcome",
    # Funciones standalone
    "detect_fractals",
    "find_control_points",
    "identify_cycles",
    "resolve_concurrences",
    # Modelos
    "Cycle",
    "ConcurrenceType",
    "DeceptionZone",
    "Direction",
    "Fractal",
    "FibonacciLevels",
    "FractalType",
    "PatternType",
    "Signal",
    "SignalType",
    "Zone",
    "ZoneType",
]
