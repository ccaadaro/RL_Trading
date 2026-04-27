#!/usr/bin/env python3
"""
scripts/test_zmq_flow.py

Offline integration test for the ZMQ pub/sub architecture.
Spins up a mock ZMQ publisher that replays real Dollar Bar data,
then verifies that the _ZmqListener thread receives and processes it.

Usage:
    python scripts/test_zmq_flow.py
"""

import sys
import json
import threading
import time
import logging
from pathlib import Path

import zmq
import pyarrow.feather as feather
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ZmqFlowTest")

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # for InstitutionalDollarStrategy

TOPIC_DOLLAR_BAR  = b"DOLLAR_BAR"
TOPIC_BOOK_TICKER = b"BOOK_TICKER"
TEST_PORT = 5599  # Use a separate port so we don't clash with a live daemon


def mock_publisher(bars: list, n_bars: int, port: int):
    """Publishes N Dollar Bars + book tickers as fast as possible."""
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    time.sleep(0.3)  # Allow SUB to connect

    logger.info("Mock publisher: sending %d Dollar Bars on port %d...", n_bars, port)
    for i, bar in enumerate(bars[:n_bars]):
        # Dollar Bar
        payload = {
            "t_open": int(bar.get("date", 0).timestamp() * 1000) if hasattr(bar.get("date", 0), "timestamp") else i,
            "t_close": i,
            "open":    float(bar.get("open", 50000)),
            "high":    float(bar.get("high", 50100)),
            "low":     float(bar.get("low",  49900)),
            "close":   float(bar.get("close", 50050)),
            "volume":  float(bar.get("volume", 40)),
            "notional": 2_000_000.0,
            "buy_volume": float(bar.get("volume", 40)) * 0.52,
            "trade_count": 1500,
            "aggressor_ratio": 0.52,
            "bar_number": i + 1,
            "book": {"best_bid": 50040.0, "best_ask": 50060.0, "mid": 50050.0,
                     "safe_buy_depth_usdt": 200_000.0, "safe_sell_depth_usdt": 180_000.0},
        }
        pub.send_multipart([TOPIC_DOLLAR_BAR, json.dumps(payload).encode()])

        # Book ticker
        book = {"best_bid": 50040.0, "best_ask": 50060.0, "mid": 50050.0, "ts": time.time()}
        pub.send_multipart([TOPIC_BOOK_TICKER, json.dumps(book).encode()])

        if (i + 1) % 10 == 0:
            logger.info("  Published bar %d/%d", i + 1, n_bars)

    logger.info("Mock publisher done.")
    pub.close()
    ctx.term()


def main():
    # Load some real Dollar Bar data as input to the mock publisher
    data_path = _HERE / "cache" / "dollar_bars_btc_2000000_regimes.feather"
    if data_path.exists():
        df = feather.read_feather(str(data_path)).head(150)
        bars = df.to_dict(orient="records")
        logger.info("Loaded %d real Dollar Bars for mock replay.", len(bars))
    else:
        logger.warning("No cached data found — using synthetic bars.")
        bars = [{"open": 50000, "high": 50100, "low": 49900, "close": 50050,
                 "volume": 40, "date": pd.Timestamp.now()}] * 150

    N_BARS = min(120, len(bars))

    # Build pipeline objects first
    from InstitutionalDollarStrategy import _ZmqListener
    import lightgbm as lgb
    from utils.risk_directors  import MahalanobisTurbulence, HMMRegimeModel
    from utils.position_sizer  import FractionalKellySizer

    model_path = _HERE / "models" / "dollar_alpha_v1" / "latest_model.txt"
    alpha = lgb.Booster(model_file=str(model_path))
    turb  = MahalanobisTurbulence(window=50, step=10)
    hmm   = HMMRegimeModel(n_components=3, n_init=3)
    sizer = FractionalKellySizer(kelly_fraction=0.5)

    import threading as _th
    signal_store = {
        "target_pos": 0.0, "regime": "unknown", "turbulence": 0.0,
        "oof_pred": 0.5, "close": None, "ts": 0.0,
        "best_bid": None, "best_ask": None, "mid": None, "n_bars": 0,
    }
    lock = _th.Lock()

    # Start subscriber FIRST (slow-joiner: SUB must connect before PUB sends)
    test_addr = f"tcp://127.0.0.1:{TEST_PORT}"
    listener = _ZmqListener(
        alpha_model=alpha, turbulence_engine=turb, hmm_model=hmm, sizer=sizer,
        signal_store=signal_store, lock=lock,
        zmq_addr=test_addr,
    )
    listener.start()
    time.sleep(0.5)  # Give subscriber time to connect and subscribe

    # Now start publisher
    pub_thread = threading.Thread(
        target=mock_publisher, args=(bars, N_BARS, TEST_PORT), daemon=True
    )
    pub_thread.start()

    # Wait for bars to flow and pipeline to fire
    deadline = time.time() + 30
    while time.time() < deadline:
        with lock:
            n = signal_store["n_bars"]
            tgt = signal_store["target_pos"]
            regime = signal_store["regime"]

        if n >= 60:
            logger.info("✓ ZMQ flow test passed: received %d bars, target_pos=%.4f, regime=%s",
                        n, tgt, regime)
            break
        logger.info("Waiting... bars_received=%d", n)
        time.sleep(2)
    else:
        logger.error("✗ ZMQ flow test FAILED: only %d bars received within 30s", n)
        sys.exit(1)

    listener.stop()
    listener.join(timeout=3)
    print("\n=== ZMQ Flow Test: ALL PASS ===")


if __name__ == "__main__":
    main()
