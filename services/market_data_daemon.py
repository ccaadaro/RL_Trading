#!/usr/bin/env python3
"""
services/market_data_daemon.py

Real-time Binance microstructure daemon.
Consumes 2 WebSocket streams simultaneously:
  - aggTrade  → Builds real Dollar Bars from individual transactions
  - bookTicker → Maintains live best bid/ask for execution pricing

When a Dollar Bar is completed (notional >= V_BAR_TARGET), serializes and
broadcasts it via ZeroMQ PUB socket on tcp://127.0.0.1:5555.

Subscribers (InstitutionalDollarStrategy, ExecutionRouter, HedgeManager)
connect as ZMQ SUB and receive complete bar dicts + book state.

Usage:
    python services/market_data_daemon.py --symbol BTCUSDT --bar-size 2000000

Keepalive: runs indefinitely. Use systemd or nohup.
"""

import asyncio
import json
import logging
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import websockets
import zmq
import zmq.asyncio

LOG_PATH = Path(__file__).parent.parent / "logs"
LOG_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH / "daemon.log"),
    ]
)
logger = logging.getLogger("MarketDataDaemon")

# ─── ZeroMQ topics ────────────────────────────────────────────────────────────
TOPIC_DOLLAR_BAR  = b"DOLLAR_BAR"
TOPIC_BOOK_TICKER = b"BOOK_TICKER"
TOPIC_ORDER_BOOK  = b"ORDER_BOOK"

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


class DollarBarAssembler:
    """
    Accumulates aggTrade messages until V_BAR_TARGET notional is reached,
    then emits a completed Dollar Bar dict.
    """
    def __init__(self, v_bar_target: float = 2_000_000.0):
        self.v_bar = v_bar_target
        self._reset()

    def _reset(self):
        self.open   = None
        self.high   = -float("inf")
        self.low    = float("inf")
        self.close  = None
        self.volume = 0.0
        self.notional = 0.0
        self.buy_volume = 0.0   # Aggressor buy volume (taker side = buyer)
        self.trade_count = 0
        self.t_open = None
        self.t_close = None
        self._imbalance_acc: list = []  # L1 book imbalance samples during this bar

    def push_imbalance(self, imbalance: float) -> None:
        """Record a book imbalance snapshot during the current bar."""
        self._imbalance_acc.append(imbalance)

    def update(self, price: float, qty: float, is_buyer_maker: bool, event_time: int) -> Optional[dict]:
        """
        Processes a single aggTrade. Returns a completed bar dict when threshold crossed,
        otherwise returns None.
        """
        if self.open is None:
            self.open   = price
            self.t_open = event_time

        self.high   = max(self.high, price)
        self.low    = min(self.low, price)
        self.close  = price
        self.volume += qty
        self.notional += price * qty
        self.trade_count += 1
        self.t_close = event_time

        # is_buyer_maker=True means the buyer is the market maker (passive), so the seller is the aggressor
        if not is_buyer_maker:
            self.buy_volume += qty  # aggressive buys

        if self.notional >= self.v_bar:
            bar = {
                "t_open":      self.t_open,
                "t_close":     event_time,
                "open":        self.open,
                "high":        self.high,
                "low":         self.low,
                "close":       self.close,
                "volume":      self.volume,
                "notional":    self.notional,
                "buy_volume":  self.buy_volume,
                "trade_count": self.trade_count,
                "aggressor_ratio": self.buy_volume / self.volume if self.volume > 0 else 0.5,
                # L1 order book imbalance averaged over bookTicker updates during this bar.
                # Range [-1, +1]: +1 = full bid dominance, -1 = full ask dominance.
                "book_imbalance": float(sum(self._imbalance_acc) / len(self._imbalance_acc))
                                  if self._imbalance_acc else 0.0,
            }
            self._reset()
            return bar

        return None


class MarketDataDaemon:
    def __init__(self, symbol: str = "btcusdt", v_bar_target: float = 2_000_000.0, zmq_port: int = 5555):
        self.symbol      = symbol.lower()
        self.assembler   = DollarBarAssembler(v_bar_target)
        self.zmq_port    = zmq_port
        self.symbol     = symbol.lower()
        self.assembler  = DollarBarAssembler(v_bar_target=v_bar_target)
        self.zmq_port   = zmq_port
        
        # State
        self.best_bid   = 0.0
        self.best_ask   = 0.0
        self.book_imbalance = 0.0
        self.l2_imbalance   = 0.0
        self.bar_count      = 0
        self._reconnect_delay = 1.0

        # ZMQ async context
        self.ctx     = zmq.asyncio.Context()
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.bind(f"tcp://127.0.0.1:{zmq_port}")
        logger.info(f"ZMQ PUB bound to tcp://127.0.0.1:{zmq_port}")

    def _build_stream_url(self) -> str:
        streams = [
            f"{self.symbol}@aggTrade",
            f"{self.symbol}@bookTicker",
            f"{self.symbol}@depth20@100ms",
        ]
        return BINANCE_WS_BASE + "/".join(streams)

    async def _publish(self, topic: bytes, payload: dict):
        msg = json.dumps(payload).encode()
        await self.pub_sock.send_multipart([topic, msg])

    async def _handle_parsed_message(self, msg: dict):
        stream = msg.get("stream", "")
        data   = msg.get("data", {})

        if "@aggTrade" in stream:
            price          = float(data["p"])
            qty            = float(data["q"])
            is_buyer_maker = data["m"]          # True = seller is aggressor
            event_time     = data["T"]

            # Snapshot current L1 + L2 imbalance into the bar being assembled
            self.assembler.push_imbalance(self.book_imbalance)
            # We also pass the L2 imbalance (this could be used for advanced Alpha later)
            
            bar = self.assembler.update(price, qty, is_buyer_maker, event_time)
            if bar is not None:
                self.bar_count += 1
                bar["bar_number"] = self.bar_count
                # Include L2 imbalance in the bar dict for the alpha model
                bar["l2_imbalance_feature"] = self.l2_imbalance
                await self._publish(TOPIC_DOLLAR_BAR, bar)
                logger.info(
                    f"Dollar Bar #{self.bar_count} | close={bar['close']:.2f} "
                    f"aggr={bar['aggressor_ratio']:.3f} "
                    f"L1_imb={bar['book_imbalance']:+.2f} L2_imb={self.l2_imbalance:+.2f}"
                )

        elif "@bookTicker" in stream:
            # L1 Imbalance from Best Bid/Ask Qty
            self.best_bid = float(data.get("b", 0))
            self.best_ask = float(data.get("a", 0))
            bid_q = float(data.get("B", 0))
            ask_q = float(data.get("A", 0))
            
            total = bid_q + ask_q
            self.book_imbalance = (bid_q - ask_q) / total if total > 0 else 0.0

            book_state = {"bid": self.best_bid, "ask": self.best_ask, "l1_imb": self.book_imbalance}
            await self._publish(TOPIC_BOOK_TICKER, book_state)

        elif "@depth" in stream:
            # L2 Imbalance from Top 10 levels
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            # Calculate notional for top 10 levels
            bid_notional = sum(float(b[0]) * float(b[1]) for b in bids[:10])
            ask_notional = sum(float(a[0]) * float(a[1]) for a in asks[:10])
            total = bid_notional + ask_notional
            self.l2_imbalance = (bid_notional - ask_notional) / total if total > 0 else 0.0
            
            l2_state = {
                "l2_imbalance": self.l2_imbalance,
                "ts":           time.time()
            }
            await self._publish(TOPIC_ORDER_BOOK, l2_state)

    async def _handle_message(self, raw: str):
        await self._handle_parsed_message(json.loads(raw))

    async def _run_stream(self):
        url = self._build_stream_url()
        logger.info(f"Connecting to {url}")
        ping_interval = 20  # Binance requires ping every 20s, disconnects at 24h

        async with websockets.connect(url, ping_interval=ping_interval) as ws:
            self._reconnect_delay = 1.0  # reset on successful connection
            # BUG-08 FIX: Reset partial bar state on each new connection.
            # Without this, the first bar after reconnect mixes trades from
            # two separate market sessions, corrupting OHLCV silently.
            self.assembler._reset()
            logger.info("WebSocket connected. Bar assembler reset for clean state.")
            async for raw in ws:
                await self._handle_message(raw)

    async def run(self):
        while True:
            try:
                await self._run_stream()
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning(f"WS disconnected: {e}. Reconnecting in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(5)


def main():
    ap = argparse.ArgumentParser(description="Binance Dollar Bar Daemon (ZeroMQ publisher)")
    ap.add_argument("--symbol",   default="BTCUSDT", help="Binance symbol, e.g. BTCUSDT")
    ap.add_argument("--bar-size", type=float, default=2_000_000.0, help="Dollar bar notional target")
    ap.add_argument("--zmq-port", type=int,   default=5555,        help="ZMQ PUB port")
    args = ap.parse_args()

    daemon = MarketDataDaemon(
        symbol=args.symbol,
        v_bar_target=args.bar_size,
        zmq_port=args.zmq_port,
    )
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
