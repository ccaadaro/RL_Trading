import threading
import pandas as pd
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    @abstractmethod
    def get_latest_data(self) -> pd.DataFrame:
        """Return the latest available OHLCV + Microstructure dataframe."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if enough data is available for inference."""
        pass

class ZmqDollarBarProvider(MarketDataProvider, threading.Thread):
    """
    Subscribes to ZMQ PUB socket for Dollar Bars.
    Maintains a thread-safe buffer compatible with InstitutionalDollarStrategy.
    """
    def __init__(self, zmq_addr: str, topic: bytes, buffer_size: int = 2000):
        super().__init__(daemon=True, name="ZmqDollarBarProvider")
        self._zmq_addr = zmq_addr
        self._topic = topic
        self._buffer_size = buffer_size
        self._bar_buffer: list[dict] = [] # Strategy expects _bar_buffer
        self._lock = threading.Lock()
        self._running = True
        
    def run(self):
        try:
            import zmq
        except ImportError:
            logger.error("pyzmq not installed. ZmqDollarBarProvider inactive.")
            return

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self._zmq_addr)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        
        logger.info(f"ZmqDollarBarProvider listening on {self._zmq_addr} for {self._topic}")
        
        while self._running:
            try:
                topic, payload_bytes = sock.recv_multipart()
                payload = json.loads(payload_bytes.decode())
                
                with self._lock:
                    self._bar_buffer.append(payload)
                    if len(self._bar_buffer) > self._buffer_size:
                        self._bar_buffer = self._bar_buffer[-(self._buffer_size//2):]
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"ZMQ Provider error: {e}")
                time.sleep(1)

    def stop(self):
        self._running = False

    def get_latest_data(self) -> pd.DataFrame:
        with self._lock:
            if not self._bar_buffer:
                return pd.DataFrame()
            return pd.DataFrame(self._bar_buffer)

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._bar_buffer) >= 50

class FreqtradeCandleProvider(MarketDataProvider):
    """
    Adapts Freqtrade's internal candle dataframe to the strategy's expected format.
    Used for 1h candles (Phase 9 pivot).
    """
    def __init__(self):
        self._latest_df = pd.DataFrame()
        self._lock = threading.Lock()

    def update(self, df: pd.DataFrame):
        with self._lock:
            self._latest_df = df.copy()

    def get_latest_data(self) -> pd.DataFrame:
        with self._lock:
            return self._latest_df

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._latest_df) >= 50
