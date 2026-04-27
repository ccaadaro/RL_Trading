"""
Patrones de entrada de la estrategia Fractal-Fibonacci.

Patrones implementados:
  1. Primer engaño (y segundo/tercer engaño como extensión)
  2. Entrada profunda en zona
  3. Engaño extremo
  4. Doble suelo / doble techo con impulso
"""

import numpy as np
import pandas as pd
from typing import Optional

from .models import (
    Cycle, DeceptionZone, Direction, FibonacciLevels, PatternType,
    Signal, SignalType, Zone, ZoneType,
)
from .cycles import compute_fibonacci

# Tolerancia de 1 pip (ajustar según el activo)
PIP_TOLERANCE = 0.0001

# Parámetros del patrón de engaño
DECEPTION_LOWER_RATIO = 1.382
DECEPTION_UPPER_RATIO = 1.618
MAX_DECEPTION_COUNT = 3

# Umbral mínimo para engaño extremo (25% de zona de indecisión)
EXTREME_DECEPTION_THRESHOLD = 0.25

# Ratio mínimo riesgo/recompensa para aceptar una señal (aplica a deep_entry, extreme_deception, double_bottom_top)
MIN_RR_RATIO = 2.0

# Ratio para cierre parcial en fibo de seguimiento
PARTIAL_CLOSE_RATIO = 0.618


def _compute_followup_fib(
    origin: float,
    extreme: float,
    direction: Direction,
) -> FibonacciLevels:
    """Fibonacci de seguimiento desde el origen del movimiento."""
    return compute_fibonacci(origin, extreme, direction)


def _deception_zone_from_fib(
    fib: FibonacciLevels,
    direction: Direction,
) -> DeceptionZone:
    """
    Calcula la zona de engaño (138.2%-161.8%) a partir de un Fibonacci de entrada.

    Para BULLISH: el fib mide la caída (origin=high, extreme=low).
      Las extensiones 138.2% y 161.8% quedan POR DEBAJO del low → zona de engaño bajista.
    Para BEARISH: el fib mide la subida (origin=low, extreme=high).
      Las extensiones 138.2% y 161.8% quedan POR ENCIMA del high → zona de engaño alcista.
    """
    size = fib.level_100 - fib.origin  # negativo para BULLISH (caída), positivo para BEARISH (subida)
    level_1382 = fib.origin + size * DECEPTION_LOWER_RATIO
    level_1618 = fib.origin + size * DECEPTION_UPPER_RATIO

    if direction == Direction.BULLISH:
        # Zona está por debajo: 161.8% es el más bajo, 138.2% el menos bajo
        return DeceptionZone(
            lower=min(level_1382, level_1618),
            upper=max(level_1382, level_1618),
            direction=direction,
        )
    else:
        # Zona está por encima: 138.2% es el menos alto, 161.8% el más alto
        return DeceptionZone(
            lower=min(level_1382, level_1618),
            upper=max(level_1382, level_1618),
            direction=direction,
        )


def detect_first_deception(
    df: pd.DataFrame,
    zone: Zone,
    cycle: Cycle,
    entry_index: int,
) -> Optional[Signal]:
    """
    Detecta y genera señal para el patrón de primer engaño.

    El precio debe estar en la zona, formar 3 pautas (llegada, rechazo,
    engaño) y el engaño debe respetar los niveles 138.2%-161.8%.

    Args:
        df: DataFrame OHLCV completo.
        zone: Zona de trabajo activa donde está el precio.
        cycle: Ciclo al que pertenece la zona.
        entry_index: Índice desde el que buscar el patrón.

    Returns:
        Signal si se detecta el patrón, None si no.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    direction = zone.direction
    deception_count = 0

    # Buscar pauta 1: precio toca la zona
    pauta1_idx = None
    for i in range(entry_index, min(entry_index + 100, n)):
        if zone.contains(closes[i]):
            pauta1_idx = i
            break

    if pauta1_idx is None:
        return None

    # Pauta 2: rechazo desde la zona — buscar extremo opuesto
    pauta2_idx = None
    pauta2_price = None
    for i in range(pauta1_idx + 1, min(pauta1_idx + 50, n)):
        if direction == Direction.BULLISH:
            if lows[i] < zone.lower:
                pauta2_idx = i
                pauta2_price = lows[i]
                break
        else:
            if highs[i] > zone.upper:
                pauta2_idx = i
                pauta2_price = highs[i]
                break

    if pauta2_idx is None:
        return None

    # Fibonacci de entrada midiendo el movimiento de engaño (desde el punto previo al extremo)
    # Para BULLISH: la manipulación empuja el precio ABAJO → medimos la caída desde el alto al bajo
    # Para BEARISH: la manipulación empuja el precio ARRIBA → medimos la subida desde el bajo al alto
    if direction == Direction.BULLISH:
        fib_entry = compute_fibonacci(highs[pauta1_idx], pauta2_price, Direction.BEARISH)
    else:
        fib_entry = compute_fibonacci(lows[pauta1_idx], pauta2_price, Direction.BULLISH)

    dec_zone = _deception_zone_from_fib(fib_entry, direction)

    # Pauta 3: precio alcanza la zona de engaño (138.2%-161.8%)
    for attempt in range(MAX_DECEPTION_COUNT):
        pauta3_idx = None
        for i in range(pauta2_idx + 1, min(pauta2_idx + 80, n)):
            price_to_check = highs[i] if direction == Direction.BEARISH else lows[i]
            if dec_zone.contains(price_to_check):
                pauta3_idx = i
                break

        if pauta3_idx is None:
            break

        deception_count += 1

        # Verificar proporcionalidad: retroceso debe tocar 38.2 mínimo
        if direction == Direction.BULLISH:
            retrace_low = lows[pauta2_idx: pauta3_idx + 1].min()
            if retrace_low > fib_entry.level_382:
                # No proporcional, buscar segundo engaño
                pauta2_idx = pauta3_idx
                pauta2_price = highs[pauta3_idx] if direction == Direction.BEARISH else lows[pauta3_idx]
                fib_entry = compute_fibonacci(pauta2_price, closes[pauta1_idx], direction)
                dec_zone = _deception_zone_from_fib(fib_entry, direction)
                continue
        else:
            retrace_high = highs[pauta2_idx: pauta3_idx + 1].max()
            if retrace_high < fib_entry.level_382:
                pauta2_idx = pauta3_idx
                pauta2_price = lows[pauta3_idx] if direction == Direction.BULLISH else highs[pauta3_idx]
                fib_entry = compute_fibonacci(pauta2_price, closes[pauta1_idx], direction)
                dec_zone = _deception_zone_from_fib(fib_entry, direction)
                continue

        # Patrón válido: calcular niveles de entrada
        # Entrada agresiva: cruzar 1 pip el nivel de engaño cuando el precio vuelve
        if direction == Direction.BULLISH:
            # Compramos cuando precio sube de vuelta por encima del 138.2% (dec_zone.upper)
            entry_price = dec_zone.upper + PIP_TOLERANCE
            # Stop por debajo del 161.8% (si rompe, el engaño falla)
            stop_price = dec_zone.lower - PIP_TOLERANCE
        else:
            # Vendemos cuando precio baja de vuelta por debajo del 138.2% (dec_zone.lower)
            entry_price = dec_zone.lower - PIP_TOLERANCE
            # Stop por encima del 161.8% (si rompe, el engaño falla)
            stop_price = dec_zone.upper + PIP_TOLERANCE

        # Target: zona contraria más alejada del ciclo, validando que esté en el lado correcto
        target = _get_target_price(cycle, zone, direction, entry_price)

        pattern_type = [
            PatternType.FIRST_DECEPTION,
            PatternType.SECOND_DECEPTION,
            PatternType.THIRD_DECEPTION,
        ][min(attempt, 2)]

        return Signal(
            index=pauta3_idx,
            signal_type=SignalType.ENTER_AGGRESSIVE,
            direction=direction,
            pattern=pattern_type,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target,
            cycle_id=cycle.cycle_id,
            zone_type=zone.zone_type,
            notes=f"Engaño #{deception_count} | Zona: {zone.zone_type.value}",
        )

    return None


def detect_deep_entry(
    df: pd.DataFrame,
    zone: Zone,
    cycle: Cycle,
    entry_index: int,
) -> Optional[Signal]:
    """
    Detecta patrón de entrada profunda: precio se pasa del 161.8% sin respetarlo
    y luego retestea.

    Returns:
        Signal si se detecta, None si no.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    direction = zone.direction
    zone_size = zone.upper - zone.lower

    # Umbral de escape profundo: mayor entre zone_size y 1% de zone.lower
    # Equivale aproximadamente al nivel 161.8% del engaño (más allá de la zona de engaño)
    deep_threshold = max(zone_size, zone.lower * 0.01)

    # Buscar primero un toque de zona (pauta 1), luego el escape profundo posterior
    zone_touch_idx = None
    for i in range(entry_index, min(entry_index + 80, n)):
        if zone.contains(closes[i]):
            zone_touch_idx = i
            break

    if zone_touch_idx is None:
        return None

    # El escape profundo debe ocurrir DESPUÉS del toque de zona
    deep_idx = None
    for i in range(zone_touch_idx + 1, min(zone_touch_idx + 80, n)):
        if direction == Direction.BULLISH and lows[i] < zone.lower - deep_threshold:
            deep_idx = i
            break
        elif direction == Direction.BEARISH and highs[i] > zone.upper + deep_threshold:
            deep_idx = i
            break

    if deep_idx is None:
        return None

    # Buscar retesteo: precio vuelve a la zona
    retest_idx = None
    for i in range(deep_idx + 1, min(deep_idx + 80, n)):
        if zone.contains(closes[i]):
            retest_idx = i
            break

    if retest_idx is None:
        return None

    # Fibonacci de seguimiento desde extremo profundo hasta borde de zona:
    # entrada en 61.8% de la recuperación esperada
    if direction == Direction.BULLISH:
        deep_price = lows[deep_idx]
        fib_followup = compute_fibonacci(deep_price, zone.upper, direction)
        entry_price = fib_followup.level_618
        stop_price = deep_price - PIP_TOLERANCE
    else:
        deep_price = highs[deep_idx]
        fib_followup = compute_fibonacci(deep_price, zone.lower, direction)
        entry_price = fib_followup.level_618
        stop_price = deep_price + PIP_TOLERANCE

    target = _get_target_price(cycle, zone, direction, entry_price)

    # Filtro mínimo R:R
    if target is not None:
        risk = abs(entry_price - stop_price)
        reward = abs(target - entry_price)
        if risk == 0 or reward / risk < MIN_RR_RATIO:
            return None

    return Signal(
        index=retest_idx,
        signal_type=SignalType.ENTER_CALM,
        direction=direction,
        pattern=PatternType.DEEP_ENTRY,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target,
        cycle_id=cycle.cycle_id,
        zone_type=zone.zone_type,
        notes="Entrada profunda en zona",
    )


def detect_extreme_deception(
    df: pd.DataFrame,
    zone: Zone,
    next_level: float,
    cycle: Cycle,
    entry_index: int,
) -> Optional[Signal]:
    """
    Detecta patrón de engaño extremo: precio escapa de la zona operable
    hacia la zona de indecisión (≥25% de su tamaño) sin llegar al siguiente
    nivel/zona, y luego reingresa.

    Args:
        next_level: Precio del siguiente nivel/zona mayor que definiría rotura.

    Returns:
        Signal si se detecta, None si no.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    direction = zone.direction
    # Para BULLISH: next_level es la resistencia POR ENCIMA de zone.upper (e.g., zona MID)
    # Para BEARISH: next_level es el soporte POR DEBAJO de zone.lower (e.g., zona MID)
    # El escape va hacia la "zona de indecisión" (MID) sin llegar al siguiente nivel
    if direction == Direction.BULLISH:
        indecision_size = abs(next_level - zone.upper)
    else:
        indecision_size = abs(next_level - zone.lower)

    if indecision_size == 0:
        return None

    # Detectar escape de zona (requiere cierre fuera de la zona, no solo mecha):
    # BULLISH: precio escapa POR ENCIMA de zone.upper (hacia zona de indecisión por arriba)
    # BEARISH: precio escapa POR DEBAJO de zone.lower (hacia zona de indecisión por abajo)
    escape_idx = None
    escape_price = None
    for i in range(entry_index, min(entry_index + 100, n)):
        if direction == Direction.BULLISH and highs[i] > zone.upper and closes[i] > zone.upper:
            escape_idx = i
            escape_price = highs[i]
            break
        elif direction == Direction.BEARISH and lows[i] < zone.lower and closes[i] < zone.lower:
            escape_idx = i
            escape_price = lows[i]
            break

    if escape_idx is None:
        return None

    # Verificar que no llegó al siguiente nivel (sería rotura real, no engaño)
    if direction == Direction.BULLISH and escape_price >= next_level:
        return None
    if direction == Direction.BEARISH and escape_price <= next_level:
        return None

    # Verificar que el escape es ≥25% de la zona de indecisión
    if direction == Direction.BULLISH:
        escape_magnitude = escape_price - zone.upper
    else:
        escape_magnitude = zone.lower - escape_price

    if escape_magnitude / indecision_size < EXTREME_DECEPTION_THRESHOLD:
        return None

    # Esperar reingreso en la zona
    reentry_idx = None
    for i in range(escape_idx + 1, min(escape_idx + 60, n)):
        if zone.contains(closes[i]):
            reentry_idx = i
            break

    if reentry_idx is None:
        return None

    # Entrada al momento del reingreso:
    # BULLISH: precio escapó arriba y volvió → compramos en el reingreso
    #   entry en zone.upper - PIP (acaba de volver al interior desde arriba)
    #   stop en el mínimo del engaño = low de la vela de escape (spec: "mínimo del engaño")
    # BEARISH: precio escapó abajo y volvió → vendemos en el reingreso
    #   entry en zone.lower + PIP (acaba de volver al interior desde abajo)
    #   stop en el máximo del engaño = high de la vela de escape (spec: "máximo del engaño")
    if direction == Direction.BULLISH:
        entry_price = zone.upper - PIP_TOLERANCE
        stop_price = lows[escape_idx] - PIP_TOLERANCE
    else:
        entry_price = zone.lower + PIP_TOLERANCE
        stop_price = highs[escape_idx] + PIP_TOLERANCE

    target = _get_target_price(cycle, zone, direction, entry_price)

    # Filtro mínimo R:R
    if target is not None:
        risk = abs(entry_price - stop_price)
        reward = abs(target - entry_price)
        if risk == 0 or reward / risk < MIN_RR_RATIO:
            return None

    return Signal(
        index=reentry_idx,
        signal_type=SignalType.ENTER_AGGRESSIVE,
        direction=direction,
        pattern=PatternType.EXTREME_DECEPTION,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target,
        cycle_id=cycle.cycle_id,
        zone_type=zone.zone_type,
        notes=f"Engaño extremo: escape {escape_magnitude:.5f} / indecisión {indecision_size:.5f}",
    )


def detect_double_bottom_top(
    df: pd.DataFrame,
    zone: Zone,
    cycle: Cycle,
    entry_index: int,
) -> Optional[Signal]:
    """
    Detecta patrón de doble suelo/doble techo con impulso.

    Condición: no hay engaños proporcionales pero el precio supera los
    máximos/mínimos de la pauta 2 con extensión ≥ 1/3 de la altura de la pauta 2.

    Returns:
        Signal si se detecta, None si no.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    direction = zone.direction

    # Pauta 1: extremo en zona
    pauta1_idx = None
    pauta1_price = None
    for i in range(entry_index, min(entry_index + 60, n)):
        if zone.contains(highs[i] if direction == Direction.BEARISH else lows[i]):
            pauta1_idx = i
            pauta1_price = lows[i] if direction == Direction.BULLISH else highs[i]
            break

    if pauta1_idx is None:
        return None

    # Pauta 2: retroceso — buscar extremo de retroceso
    pauta2_idx = None
    pauta2_price = None
    for i in range(pauta1_idx + 1, min(pauta1_idx + 60, n)):
        if direction == Direction.BULLISH:
            if highs[i] > zone.upper:
                pauta2_idx = i
                pauta2_price = highs[i]
                break
        else:
            if lows[i] < zone.lower:
                pauta2_idx = i
                pauta2_price = lows[i]
                break

    if pauta2_idx is None:
        return None

    pauta2_height = abs(pauta2_price - pauta1_price)
    if pauta2_height == 0:
        return None

    threshold = pauta2_height / 3.0

    # Buscar superación de máximos/mínimos de pauta 2 con extensión ≥ 1/3
    impulse_idx = None
    max_dilatation = None
    for i in range(pauta2_idx + 1, min(pauta2_idx + 80, n)):
        if direction == Direction.BULLISH:
            if highs[i] > pauta2_price + threshold:
                impulse_idx = i
                max_dilatation = highs[i: min(i + 30, n)].max()
                break
        else:
            if lows[i] < pauta2_price - threshold:
                impulse_idx = i
                max_dilatation = lows[i: min(i + 30, n)].min()
                break

    if impulse_idx is None or max_dilatation is None:
        return None

    # Fibonacci desde pauta1 (extremo) a max_dilatation; entrada calmada en 61.8%
    # BULLISH: origen=pauta1_price(bajo), extremo=max_dilatation(alto)
    #   entry en 61.8% (retroceso del impulso, zona de soporte natural)
    #   stop por debajo del primer mínimo (invalidación del doble suelo)
    # BEARISH: origen=pauta1_price(alto), extremo=max_dilatation(bajo)
    #   entry en 61.8% (retroceso, zona de resistencia natural)
    #   stop por encima del primer máximo (invalidación del doble techo)
    fib = compute_fibonacci(pauta1_price, max_dilatation, direction)
    if direction == Direction.BULLISH:
        entry_price = fib.level_618
        stop_price = pauta1_price - PIP_TOLERANCE
    else:
        entry_price = fib.level_618
        stop_price = pauta1_price + PIP_TOLERANCE

    # Target: zona contraria más alejada del ciclo (por encima de max_dilatation para BULLISH)
    target = _get_target_price(cycle, zone, direction, entry_price)
    if target is None:
        return None  # Sin target válido no hay R:R definido

    # Filtro R:R mínimo 1.5:1 (más permisivo que otros patrones — entry es en zona de soporte)
    risk = abs(entry_price - stop_price)
    reward = abs(target - entry_price)
    if risk == 0 or reward / risk < 1.5:
        return None

    return Signal(
        index=impulse_idx,
        signal_type=SignalType.ENTER_CALM,
        direction=direction,
        pattern=PatternType.DOUBLE_BOTTOM_TOP,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target,
        cycle_id=cycle.cycle_id,
        zone_type=zone.zone_type,
        notes=f"Doble {'suelo' if direction == Direction.BULLISH else 'techo'} con impulso",
    )


def _get_target_price(
    cycle: Cycle,
    zone: Zone,
    direction: Direction,
    entry_price: Optional[float] = None,
) -> Optional[float]:
    """
    Obtiene el target de la operación: zona opuesta más alejada del ciclo.
    Para BULLISH: target es el borde inferior de la zona más alta (por encima de entrada).
    Para BEARISH: target es el borde superior de la zona más baja (por debajo de entrada).

    Si se pasa entry_price, se garantiza que el target está en el lado correcto.
    """
    if not cycle.zones:
        return None
    valid = [z for z in cycle.zones if z.valid and z.zone_type != zone.zone_type]
    if not valid:
        return None
    if direction == Direction.BULLISH:
        candidates = sorted(valid, key=lambda z: z.lower, reverse=True)
        if entry_price is not None:
            candidates = [z for z in candidates if z.lower > entry_price]
        return candidates[0].lower if candidates else None
    else:
        candidates = sorted(valid, key=lambda z: z.upper)
        if entry_price is not None:
            candidates = [z for z in candidates if z.upper < entry_price]
        return candidates[0].upper if candidates else None
