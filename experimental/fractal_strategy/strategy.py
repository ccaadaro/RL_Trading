"""
Orquestador principal de la estrategia Fractal-Fibonacci.

Uso básico:
    from fractal_strategy import FractalStrategy
    import pandas as pd

    df = pd.read_csv("ohlcv.csv")   # columnas: open, high, low, close, volume
    strategy = FractalStrategy()
    result = strategy.run(df)

    print(result["signals"])        # lista de señales
    print(result["cycles"])         # ciclos identificados
    print(result["zones"])          # zonas de trabajo válidas
"""

import pandas as pd
from typing import Any

from .concurrences import resolve_concurrences
from .cycles import get_active_cycles, identify_cycles
from .fractals import detect_fractals, find_control_points
from .models import Cycle, Direction, Fractal, Signal, Zone
from .patterns import (
    detect_deep_entry,
    detect_double_bottom_top,
    detect_extreme_deception,
    detect_first_deception,
)

REQUIRED_COLUMNS = {"high", "low", "close"}


class FractalStrategy:
    """
    Implementación de la estrategia de trading Fractal-Fibonacci de Joaquín Vega.

    Parámetros:
        min_impulse_pips: Tamaño mínimo del impulso para identificar un ciclo.
            Ajustar según el activo y temporalidad (ej. 0.001 para forex,
            100 para índices).
        pip_tolerance: Tolerancia en precio para entradas (1 pip del activo).
        scan_lookback: Número máximo de velas a mirar atrás al buscar patrones.
    """

    def __init__(
        self,
        min_impulse_pips: float = 0.0,
        pip_tolerance: float = 0.0001,
        scan_lookback: int = 100,
    ) -> None:
        self.min_impulse_pips = min_impulse_pips
        self.pip_tolerance = pip_tolerance
        self.scan_lookback = scan_lookback

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Ejecuta el análisis completo sobre un DataFrame OHLCV.

        Args:
            df: DataFrame con columnas 'open', 'high', 'low', 'close' y
                opcionalmente 'volume'. El índice puede ser numérico o datetime.

        Returns:
            Diccionario con:
              - "fractals":       list[Fractal]
              - "control_points": list[ControlPoint]
              - "cycles":         list[Cycle]
              - "zones":          list[Zone]  (ya resueltas concurrencias)
              - "signals":        list[Signal]
              - "signals_df":     pd.DataFrame con las señales
        """
        self._validate(df)
        df = df.reset_index(drop=True)

        # 1. Detectar fractales
        fractals = detect_fractals(df)

        # 2. Puntos de control validados
        control_points = find_control_points(df, fractals)

        # 3. Identificar ciclos operables y calcular zonas
        cycles = identify_cycles(df, fractals, self.min_impulse_pips)

        # 4. Resolver concurrencias entre zonas
        all_zones = [z for c in cycles for z in c.zones]
        current_price = float(df["close"].iloc[-1])
        valid_zones = resolve_concurrences(all_zones, current_price)

        # 5. Buscar patrones de entrada en cada zona válida
        signals = self._scan_patterns(df, cycles, valid_zones)

        return {
            "fractals": fractals,
            "control_points": control_points,
            "cycles": cycles,
            "zones": valid_zones,
            "signals": signals,
            "signals_df": self._signals_to_df(signals),
        }

    def _scan_patterns(
        self,
        df: pd.DataFrame,
        cycles: list[Cycle],
        zones: list[Zone],
    ) -> list[Signal]:
        """Busca patrones de entrada en cada zona válida de cada ciclo activo."""
        signals: list[Signal] = []
        n = len(df)
        highs = df["high"].values
        lows = df["low"].values

        for zone in zones:
            # Recuperar el ciclo al que pertenece la zona
            cycle = next((c for c in cycles if c.cycle_id == zone.cycle_id), None)
            if cycle is None or not cycle.active:
                continue

            # Índice de inicio de búsqueda: desde que el ciclo está confirmado
            start_idx = max(0, cycle.retrace_index - self.scan_lookback)

            # Buscar el nivel siguiente para engaño extremo
            next_level = self._get_next_level(zone, zones, cycle)

            # Intentar todos los patrones; usar el primero que detecte señal
            for detector in [
                lambda: detect_first_deception(df, zone, cycle, start_idx),
                lambda: detect_deep_entry(df, zone, cycle, start_idx),
                lambda: detect_extreme_deception(df, zone, next_level, cycle, start_idx),
                lambda: detect_double_bottom_top(df, zone, cycle, start_idx),
            ]:
                signal = detector()
                if signal is not None:
                    signals.append(signal)
                    break  # Solo una señal por zona

        signals.sort(key=lambda s: s.index)
        return signals

    def _get_next_level(
        self,
        zone: Zone,
        all_zones: list[Zone],
        cycle: Cycle,
    ) -> float:
        """
        Determina el precio del siguiente nivel/zona mayor que definiría
        una rotura, usado para calcular la zona de indecisión del engaño extremo.
        """
        fib = cycle.fib
        if zone.direction == Direction.BULLISH:
            # Siguiente nivel por encima de la zona
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

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame debe tener columnas: {missing}")
        if len(df) < 10:
            raise ValueError("Se necesitan al menos 10 velas para el análisis.")

    @staticmethod
    def _signals_to_df(signals: list[Signal]) -> pd.DataFrame:
        if not signals:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in signals])
