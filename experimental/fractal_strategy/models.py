"""
Modelos de datos de la estrategia Fractal-Fibonacci.
Todas las estructuras usan dataclasses estándar para máxima compatibilidad.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FractalType(Enum):
    HIGH = "high"
    LOW = "low"


class ZoneType(Enum):
    HIGH = "high"      # Zona alta del ciclo (extensión 138.2)
    MID = "mid"        # Zona 61.8 (zona primaria)
    LOW = "low"        # Zona baja (retroceso 38.2 hacia 0)


class SignalType(Enum):
    ENTER_AGGRESSIVE = "enter_aggressive"   # Entrada agresiva (sin esperar cierre)
    ENTER_CALM = "enter_calm"               # Entrada calmada (en 61.8 de fibo seguimiento)
    CANCEL = "cancel"                       # Anulación del patrón
    CLOSE_HALF = "close_half"              # Cerrar 50% de posición


class PatternType(Enum):
    FIRST_DECEPTION = "first_deception"
    SECOND_DECEPTION = "second_deception"
    THIRD_DECEPTION = "third_deception"
    DEEP_ENTRY = "deep_entry"
    EXTREME_DECEPTION = "extreme_deception"
    DOUBLE_BOTTOM_TOP = "double_bottom_top"


class ConcurrenceType(Enum):
    TYPE_1 = 1   # Zona menor inmersa en mayor → eliminar menor
    TYPE_2 = 2   # Precio ataca primero zona menor → evaluar 50% libre
    TYPE_3 = 3   # Precio ataca primero zona mayor (menor en sandwich) → eliminar menor


@dataclass
class Fractal:
    """Fractal confirmado (Williams 5 velas)."""
    index: int                  # Índice de la vela central (máximo/mínimo)
    fractal_type: FractalType   # HIGH o LOW
    price: float                # Precio del máximo o mínimo
    confirmed_at: int           # Índice de la vela en que se confirmó (index + 2)
    direction: Direction        # Dirección del impulso que lo generó


@dataclass
class FibonacciLevels:
    """Niveles Fibonacci de un ciclo."""
    origin: float       # Nivel 0 (inicio del impulso)
    level_382: float    # 38.2% — activación del ciclo
    level_618: float    # 61.8% — zona primaria
    level_100: float    # 100% — extremo del impulso
    ext_neg382: float   # Extensión -38.2% (por debajo del origen en alcista)
    ext_1382: float     # Extensión 138.2% (por encima del extremo en alcista)

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "382": self.level_382,
            "618": self.level_618,
            "100": self.level_100,
            "ext_neg382": self.ext_neg382,
            "ext_1382": self.ext_1382,
        }


@dataclass
class Zone:
    """Zona de trabajo dentro de un ciclo."""
    cycle_id: int
    zone_type: ZoneType
    direction: Direction    # Dirección operativa de la zona
    upper: float            # Límite superior de la zona
    lower: float            # Límite inferior de la zona
    valid: bool = True      # False si fue eliminada por concurrencia

    @property
    def size(self) -> float:
        return abs(self.upper - self.lower)

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def free_ratio(self, other: "Zone") -> float:
        """Fracción de esta zona que está libre (no solapada) respecto a otra."""
        overlap_upper = min(self.upper, other.upper)
        overlap_lower = max(self.lower, other.lower)
        overlap = max(0.0, overlap_upper - overlap_lower)
        return 1.0 - (overlap / self.size) if self.size > 0 else 0.0


@dataclass
class Cycle:
    """Ciclo operable: fractal cuyo retroceso ≥ 38.2% de Fibonacci."""
    cycle_id: int
    direction: Direction
    impulse_start: float    # Precio inicio del impulso
    impulse_end: float      # Precio fin del impulso (extremo)
    retrace_low: float      # Precio más bajo del retroceso (alcista) o más alto (bajista)
    start_index: int        # Índice inicio impulso en el DataFrame
    end_index: int          # Índice fin impulso
    retrace_index: int      # Índice donde se produce el retroceso
    fib: FibonacciLevels = field(default=None)
    zones: list[Zone] = field(default_factory=list)
    active: bool = True

    @property
    def impulse_size(self) -> float:
        return abs(self.impulse_end - self.impulse_start)

    @property
    def retrace_depth(self) -> float:
        """Profundidad del retroceso como fracción del impulso (0.0 a 1.0+)."""
        if self.impulse_size == 0:
            return 0.0
        if self.direction == Direction.BULLISH:
            return (self.impulse_end - self.retrace_low) / self.impulse_size
        else:
            return (self.retrace_low - self.impulse_end) / self.impulse_size


@dataclass
class ControlPoint:
    """Punto de control validado (rotura correcta ≥ 1/3 del retroceso)."""
    index: int
    price: float
    direction: Direction    # Dirección de la rotura
    retrace_height: float   # Altura del retroceso que validó este punto
    threshold: float        # Umbral mínimo (1/3 de retrace_height)


@dataclass
class DeceptionZone:
    """Zona de engaño: niveles 138.2% y 161.8% del Fibonacci de entrada."""
    upper: float    # 161.8%
    lower: float    # 138.2%
    direction: Direction

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass
class Signal:
    """Señal de trading generada por la estrategia."""
    index: int                          # Índice en el DataFrame donde se genera
    signal_type: SignalType
    direction: Direction                # Dirección de la operación
    pattern: PatternType
    entry_price: float
    stop_price: float
    target_price: Optional[float] = None
    cycle_id: Optional[int] = None
    zone_type: Optional[ZoneType] = None
    notes: str = ""

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def reward(self) -> Optional[float]:
        if self.target_price is None:
            return None
        return abs(self.target_price - self.entry_price)

    @property
    def rr_ratio(self) -> Optional[float]:
        if self.reward is None or self.risk == 0:
            return None
        return self.reward / self.risk

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "pattern": self.pattern.value,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "cycle_id": self.cycle_id,
            "zone_type": self.zone_type.value if self.zone_type else None,
            "risk": self.risk,
            "reward": self.reward,
            "rr_ratio": self.rr_ratio,
            "notes": self.notes,
        }
