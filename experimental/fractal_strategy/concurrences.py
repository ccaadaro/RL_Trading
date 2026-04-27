"""
Análisis de concurrencias entre zonas de distintos ciclos.

Tipos de concurrencia (entre zonas del mismo sentido operativo):
  Tipo 1: Zona menor completamente inmersa en zona mayor → eliminar zona menor
  Tipo 2: Precio ataca primero la zona menor
           - Si zona menor tiene ≥50% libre → mantener (acotada al límite de la mayor)
           - Si zona menor tiene <50% libre → eliminar zona menor
  Tipo 3: Precio ataca primero la zona mayor (la menor queda en sandwich) → eliminar zona menor
"""

from .models import ConcurrenceType, Direction, Zone


def zones_overlap(a: Zone, b: Zone) -> bool:
    """Devuelve True si dos zonas se solapan."""
    return a.lower < b.upper and b.lower < a.upper


def classify_concurrence(
    zone_a: Zone,
    zone_b: Zone,
    current_price: float,
    direction: Direction,
) -> ConcurrenceType:
    """
    Clasifica el tipo de concurrencia entre dos zonas del mismo sentido.

    Args:
        zone_a: Primera zona.
        zone_b: Segunda zona.
        current_price: Precio actual del mercado.
        direction: Dirección operativa de las zonas (BULLISH → precio viene de abajo).

    Returns:
        ConcurrenceType (1, 2 o 3).
    """
    # Determinar cuál es la zona mayor (más grande)
    if zone_a.size >= zone_b.size:
        major, minor = zone_a, zone_b
    else:
        major, minor = zone_b, zone_a

    # Tipo 1: menor completamente dentro de mayor
    if minor.lower >= major.lower and minor.upper <= major.upper:
        return ConcurrenceType.TYPE_1

    # El precio se acerca desde fuera: determinar cuál toca primero
    if direction == Direction.BULLISH:
        # Precio viene de abajo → toca primero la zona con lower más bajo
        hits_minor_first = minor.lower <= major.lower
    else:
        # Precio viene de arriba → toca primero la zona con upper más alto
        hits_minor_first = minor.upper >= major.upper

    if hits_minor_first:
        return ConcurrenceType.TYPE_2
    else:
        return ConcurrenceType.TYPE_3


def resolve_concurrences(zones: list[Zone], current_price: float) -> list[Zone]:
    """
    Aplica las reglas de concurrencia a una lista de zonas y devuelve
    las zonas válidas tras eliminar/acotar las afectadas.

    Solo analiza pares de zonas con el mismo sentido operativo.

    Args:
        zones: Lista de zonas activas de todos los ciclos.
        current_price: Precio actual del mercado.

    Returns:
        Lista de zonas válidas (sin modificar las originales).
    """
    # Resetear validez de todas las zonas antes de re-evaluar
    for z in zones:
        z.valid = True
    valid_zones = list(zones)

    # Agrupar por dirección operativa
    for direction in (Direction.BULLISH, Direction.BEARISH):
        dir_zones = [z for z in valid_zones if z.direction == direction]

        # Comparar todos los pares
        i = 0
        while i < len(dir_zones):
            j = i + 1
            while j < len(dir_zones):
                za = dir_zones[i]
                zb = dir_zones[j]

                if not (za.valid and zb.valid):
                    j += 1
                    continue

                if not zones_overlap(za, zb):
                    j += 1
                    continue

                concurrence = classify_concurrence(za, zb, current_price, direction)
                major = za if za.size >= zb.size else zb
                minor = zb if za.size >= zb.size else za

                if concurrence == ConcurrenceType.TYPE_1:
                    # Eliminar zona menor sin evaluación
                    minor.valid = False

                elif concurrence == ConcurrenceType.TYPE_2:
                    free_ratio = minor.free_ratio(major)
                    if free_ratio >= 0.5:
                        # Acotar zona menor hasta límite de la mayor
                        if direction == Direction.BULLISH:
                            minor.upper = min(minor.upper, major.lower)
                        else:
                            minor.lower = max(minor.lower, major.upper)
                    else:
                        minor.valid = False

                elif concurrence == ConcurrenceType.TYPE_3:
                    # Sandwich: eliminar zona menor
                    minor.valid = False

                j += 1
            i += 1

    return [z for z in valid_zones if z.valid]


def find_concurrences(zones: list[Zone], current_price: float) -> list[dict]:
    """
    Detecta y describe las concurrencias presentes entre zonas, sin resolverlas.
    Útil para diagnóstico y visualización.

    Returns:
        Lista de dicts con información de cada concurrencia detectada.
    """
    result = []
    for i, za in enumerate(zones):
        for j, zb in enumerate(zones):
            if i >= j:
                continue
            if za.direction != zb.direction:
                continue
            if not zones_overlap(za, zb):
                continue
            concurrence = classify_concurrence(za, zb, current_price, za.direction)
            result.append({
                "zone_a": {"cycle_id": za.cycle_id, "type": za.zone_type.value},
                "zone_b": {"cycle_id": zb.cycle_id, "type": zb.zone_type.value},
                "concurrence_type": concurrence.value,
                "overlap_upper": min(za.upper, zb.upper),
                "overlap_lower": max(za.lower, zb.lower),
            })
    return result
