"""
Identificación de ciclos operables y cálculo de zonas Fibonacci.

Un ciclo es operable cuando el retroceso posterior al impulso toca o supera
el nivel 38.2% de Fibonacci.

Niveles Fibonacci utilizados: 0, 38.2, 61.8, 100, -38.2 (ext. negativa), 138.2 (ext. positiva)
"""

import numpy as np
import pandas as pd
from typing import Optional

from .models import (
    Cycle, Direction, FibonacciLevels, Fractal, FractalType, Zone, ZoneType
)

# Niveles Fibonacci globales
FIB_LEVELS = {
    "0": 0.0,
    "382": 0.382,
    "618": 0.618,
    "100": 1.0,
    "ext_neg382": -0.382,
    "ext_1382": 1.382,
}

# Umbral mínimo de retroceso para ciclo operable
MIN_RETRACE_FOR_CYCLE = 0.382


def compute_fibonacci(
    origin: float,
    extreme: float,
    direction: Direction,
) -> FibonacciLevels:
    """
    Calcula los 6 niveles Fibonacci para un impulso dado.

    Args:
        origin: Precio inicio del impulso (nivel 0).
        extreme: Precio fin del impulso (nivel 100).
        direction: Dirección del impulso.

    Returns:
        FibonacciLevels con los 6 niveles calculados.
    """
    size = extreme - origin  # Positivo para alcista, negativo para bajista

    def level(ratio: float) -> float:
        return origin + size * ratio

    return FibonacciLevels(
        origin=origin,
        level_382=level(FIB_LEVELS["382"]),
        level_618=level(FIB_LEVELS["618"]),
        level_100=extreme,
        ext_neg382=level(FIB_LEVELS["ext_neg382"]),
        ext_1382=level(FIB_LEVELS["ext_1382"]),
    )


def compute_zones(cycle: Cycle) -> list[Zone]:
    """
    Calcula las 3 zonas de trabajo de un ciclo operable.

    Zona alta  (HIGH): entre ext_1382 y level_100 (zona de venta en alcista)
    Zona media (MID):  alrededor del level_618
    Zona baja  (LOW):  entre level_382 y origin (zona de compra en alcista)

    Args:
        cycle: Ciclo con Fibonacci ya calculado.

    Returns:
        Lista de 3 Zone.
    """
    fib = cycle.fib
    zones = []

    if cycle.direction == Direction.BULLISH:
        # Zona alta: entre level_100 y ext_1382 → operativa bajista (venta)
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.HIGH,
            direction=Direction.BEARISH,
            upper=fib.ext_1382,
            lower=fib.level_100,
        ))
        # Zona media: alrededor del 61.8 (±19.1% del impulso, es decir 50% del tramo 38.2-100)
        zone_half = abs(fib.level_100 - fib.level_382) * 0.5
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.MID,
            direction=Direction.BULLISH,
            upper=fib.level_618 + zone_half * 0.5,
            lower=fib.level_618 - zone_half * 0.5,
        ))
        # Zona baja: entre ext_neg382 y origin → operativa alcista (compra)
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.LOW,
            direction=Direction.BULLISH,
            upper=fib.origin,
            lower=fib.ext_neg382,
        ))
    else:
        # Bearish: todo invertido
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.HIGH,
            direction=Direction.BULLISH,
            upper=fib.level_100,
            lower=fib.ext_1382,
        ))
        zone_half = abs(fib.level_382 - fib.level_100) * 0.5
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.MID,
            direction=Direction.BEARISH,
            upper=fib.level_618 + zone_half * 0.5,
            lower=fib.level_618 - zone_half * 0.5,
        ))
        zones.append(Zone(
            cycle_id=cycle.cycle_id,
            zone_type=ZoneType.LOW,
            direction=Direction.BEARISH,
            upper=fib.ext_neg382,
            lower=fib.origin,
        ))

    return zones


def identify_cycles(
    df: pd.DataFrame,
    fractals: list[Fractal],
    min_impulse_pips: float = 0.0,
) -> list[Cycle]:
    """
    Identifica los ciclos operables a partir de pares impulso-retroceso.

    Para cada par (mínimo fractal → máximo fractal) en alcista y
    (máximo fractal → mínimo fractal) en bajista, verifica si el retroceso
    posterior alcanza o supera el nivel 38.2%.

    Args:
        df: DataFrame con columnas 'high', 'low', 'close'.
        fractals: Lista de fractales confirmados.
        min_impulse_pips: Tamaño mínimo del impulso para considerar el ciclo.

    Returns:
        Lista de Cycle operables con Fibonacci y zonas calculados.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    cycles: list[Cycle] = []
    cycle_id = 0

    high_fractals = sorted(
        [f for f in fractals if f.fractal_type == FractalType.HIGH],
        key=lambda f: f.index,
    )
    low_fractals = sorted(
        [f for f in fractals if f.fractal_type == FractalType.LOW],
        key=lambda f: f.index,
    )

    # --- Ciclos alcistas: desde mínimo fractal hasta máximo fractal ---
    for low_f in low_fractals:
        # Buscar el primer máximo fractal posterior
        next_highs = [f for f in high_fractals if f.index > low_f.index]
        if not next_highs:
            continue
        high_f = next_highs[0]

        impulse_start = low_f.price
        impulse_end = high_f.price
        impulse_size = impulse_end - impulse_start

        if impulse_size < min_impulse_pips or impulse_size <= 0:
            continue

        # Buscar el mínimo del retroceso posterior al máximo
        retrace_start = high_f.confirmed_at
        retrace_end = min(retrace_start + 200, n)
        if retrace_start >= n:
            continue

        retrace_segment = lows[retrace_start:retrace_end]
        if len(retrace_segment) == 0:
            continue

        retrace_low = retrace_segment.min()
        retrace_depth = (impulse_end - retrace_low) / impulse_size

        if retrace_depth < MIN_RETRACE_FOR_CYCLE:
            continue

        retrace_idx = retrace_start + int(np.argmin(retrace_segment))

        fib = compute_fibonacci(impulse_start, impulse_end, Direction.BULLISH)
        c = Cycle(
            cycle_id=cycle_id,
            direction=Direction.BULLISH,
            impulse_start=impulse_start,
            impulse_end=impulse_end,
            retrace_low=retrace_low,
            start_index=low_f.index,
            end_index=high_f.index,
            retrace_index=retrace_idx,
            fib=fib,
        )
        c.zones = compute_zones(c)
        cycles.append(c)
        cycle_id += 1

    # --- Ciclos bajistas: desde máximo fractal hasta mínimo fractal ---
    for high_f in high_fractals:
        next_lows = [f for f in low_fractals if f.index > high_f.index]
        if not next_lows:
            continue
        low_f = next_lows[0]

        impulse_start = high_f.price
        impulse_end = low_f.price
        impulse_size = impulse_start - impulse_end

        if impulse_size < min_impulse_pips or impulse_size <= 0:
            continue

        retrace_start = low_f.confirmed_at
        retrace_end = min(retrace_start + 200, n)
        if retrace_start >= n:
            continue

        retrace_segment = highs[retrace_start:retrace_end]
        if len(retrace_segment) == 0:
            continue

        retrace_high = retrace_segment.max()
        retrace_depth = (retrace_high - impulse_end) / impulse_size

        if retrace_depth < MIN_RETRACE_FOR_CYCLE:
            continue

        retrace_idx = retrace_start + int(np.argmax(retrace_segment))

        fib = compute_fibonacci(impulse_start, impulse_end, Direction.BEARISH)
        c = Cycle(
            cycle_id=cycle_id,
            direction=Direction.BEARISH,
            impulse_start=impulse_start,
            impulse_end=impulse_end,
            retrace_low=retrace_high,
            start_index=high_f.index,
            end_index=low_f.index,
            retrace_index=retrace_idx,
            fib=fib,
        )
        c.zones = compute_zones(c)
        cycles.append(c)
        cycle_id += 1

    cycles.sort(key=lambda c: c.start_index)
    return cycles


def get_active_cycles(cycles: list[Cycle], at_index: int) -> list[Cycle]:
    """Devuelve los ciclos activos en un índice dado."""
    return [c for c in cycles if c.active and c.retrace_index <= at_index]


def deactivate_evolved_cycles(
    cycles: list[Cycle],
    current_price: float,
    at_index: int,
) -> None:
    """
    Desactiva ciclos secundarios cuando el precio toca el nivel 38.2
    del ciclo mayor (evolución de ciclo).
    Modifica la lista in-place.
    """
    active = get_active_cycles(cycles, at_index)
    for c in active:
        fib = c.fib
        if c.direction == Direction.BULLISH and current_price <= fib.level_382:
            c.active = False
        elif c.direction == Direction.BEARISH and current_price >= fib.level_382:
            c.active = False
