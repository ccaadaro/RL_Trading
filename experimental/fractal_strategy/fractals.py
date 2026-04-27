"""
Detección de fractales (Williams 5 velas) y validación de roturas.

Reglas:
- Máximo fractal: vela[i].high es el más alto de las 5 velas (2 izq + central + 2 der)
- Mínimo fractal: vela[i].low es el más bajo de las 5 velas
- Se confirma cuando cierra la vela[i+2] (retraso de 2 velas)
- Rotura correcta: el precio supera el extremo anterior por al menos 1/3 de la
  altura del retroceso actual
"""

import numpy as np
import pandas as pd
from typing import Optional

from .models import (
    ControlPoint, Direction, Fractal, FractalType
)


def detect_fractals(df: pd.DataFrame) -> list[Fractal]:
    """
    Detecta todos los fractales confirmados en el DataFrame.

    Args:
        df: DataFrame con columnas 'high' y 'low' (índice numérico o datetime).

    Returns:
        Lista de Fractal ordenada por índice de confirmación.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    fractals: list[Fractal] = []

    for i in range(2, n - 2):
        # Máximo fractal: vela i tiene el high más alto de las 5
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2]
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            fractals.append(Fractal(
                index=i,
                fractal_type=FractalType.HIGH,
                price=highs[i],
                confirmed_at=i + 2,
                direction=Direction.BEARISH,  # Máximo → impulso previo era alcista
            ))

        # Mínimo fractal: vela i tiene el low más bajo de las 5
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2]
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            fractals.append(Fractal(
                index=i,
                fractal_type=FractalType.LOW,
                price=lows[i],
                confirmed_at=i + 2,
                direction=Direction.BULLISH,  # Mínimo → impulso previo era bajista
            ))

    fractals.sort(key=lambda f: f.confirmed_at)
    return fractals


def validate_breakout(
    price: float,
    prev_extreme: float,
    retrace_height: float,
    direction: Direction,
) -> bool:
    """
    Valida si una rotura es correcta (supera el extremo anterior en al menos
    1/3 de la altura del retroceso).

    Args:
        price: Precio actual.
        prev_extreme: Máximo anterior (rotura alcista) o mínimo anterior (bajista).
        retrace_height: Altura del retroceso que formó el punto de control.
        direction: Dirección de la rotura.

    Returns:
        True si la rotura es válida.
    """
    threshold = retrace_height / 3.0
    if direction == Direction.BULLISH:
        return (price - prev_extreme) >= threshold
    else:
        return (prev_extreme - price) >= threshold


def find_control_points(
    df: pd.DataFrame,
    fractals: list[Fractal],
) -> list[ControlPoint]:
    """
    Encuentra los puntos de control validados a partir de los fractales.
    Un punto de control se valida cuando el precio supera el extremo anterior
    por al menos 1/3 de la altura del retroceso.

    Args:
        df: DataFrame con columnas 'high' y 'low'.
        fractals: Lista de fractales confirmados.

    Returns:
        Lista de ControlPoint validados.
    """
    highs = df["high"].values
    lows = df["low"].values
    control_points: list[ControlPoint] = []

    high_fractals = [f for f in fractals if f.fractal_type == FractalType.HIGH]
    low_fractals = [f for f in fractals if f.fractal_type == FractalType.LOW]

    # Roturas alcistas: precio supera un máximo fractal anterior
    for i, frac in enumerate(high_fractals[:-1]):
        next_frac = high_fractals[i + 1]
        # Buscar el mínimo entre los dos máximos fractales (retroceso)
        segment_lows = lows[frac.index: next_frac.index + 1]
        if len(segment_lows) == 0:
            continue
        retrace_low = segment_lows.min()
        retrace_height = frac.price - retrace_low

        # Verificar si el siguiente máximo supera el anterior en al menos 1/3
        if validate_breakout(next_frac.price, frac.price, retrace_height, Direction.BULLISH):
            control_points.append(ControlPoint(
                index=next_frac.confirmed_at,
                price=frac.price,
                direction=Direction.BULLISH,
                retrace_height=retrace_height,
                threshold=retrace_height / 3.0,
            ))

    # Roturas bajistas: precio supera un mínimo fractal anterior
    for i, frac in enumerate(low_fractals[:-1]):
        next_frac = low_fractals[i + 1]
        segment_highs = highs[frac.index: next_frac.index + 1]
        if len(segment_highs) == 0:
            continue
        retrace_high = segment_highs.max()
        retrace_height = retrace_high - frac.price

        if validate_breakout(next_frac.price, frac.price, retrace_height, Direction.BEARISH):
            control_points.append(ControlPoint(
                index=next_frac.confirmed_at,
                price=frac.price,
                direction=Direction.BEARISH,
                retrace_height=retrace_height,
                threshold=retrace_height / 3.0,
            ))

    control_points.sort(key=lambda cp: cp.index)
    return control_points


def get_fractal_at(
    fractals: list[Fractal],
    fractal_type: FractalType,
    before_index: int,
) -> Optional[Fractal]:
    """Devuelve el fractal confirmado más reciente de un tipo dado antes de un índice."""
    candidates = [
        f for f in fractals
        if f.fractal_type == fractal_type and f.confirmed_at <= before_index
    ]
    return candidates[-1] if candidates else None
