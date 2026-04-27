#!/usr/bin/env python3
"""
services/dashboard_server.py

Real-time dashboard backend.
Bridges ZMQ topics → WebSocket → browser.

Sources:
  tcp://127.0.0.1:5555  DOLLAR_BAR, BOOK_TICKER  (market_data_daemon)
  tcp://127.0.0.1:5556  DIAGNOSTICS              (InstitutionalDollarStrategy)

Usage:
  pip install fastapi uvicorn pyzmq
  python services/dashboard_server.py
  open http://localhost:8050
"""

import asyncio
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Set

import zmq
import zmq.asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DashboardServer")

# ── State ─────────────────────────────────────────────────────────────────────

BAR_BUFFER_LEN  = 200   # rolling window of dollar bars sent to client
DIAG_BUFFER_LEN = 500   # rolling window of diagnostics (for sparklines)

bar_buffer:   deque = deque(maxlen=BAR_BUFFER_LEN)
diag_buffer:  deque = deque(maxlen=DIAG_BUFFER_LEN)
book_state:   dict  = {}
latest_diag:  dict  = {}

clients: Set[WebSocket] = set()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Institutional Dashboard")

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


@app.get("/")
async def index():
    if _DASHBOARD_HTML.exists():
        return FileResponse(_DASHBOARD_HTML, media_type="text/html")
    return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    logger.info("Client connected (%d total)", len(clients))

    # Send full state snapshot on connect so the client renders immediately
    snapshot = {
        "type":       "snapshot",
        "bars":       list(bar_buffer),
        "diag":       list(diag_buffer),
        "book":       book_state,
        "latest_diag": latest_diag,
    }
    try:
        await ws.send_text(json.dumps(snapshot, default=str))
    except Exception:
        clients.discard(ws)
        return

    try:
        while True:
            await asyncio.sleep(60)   # keep-alive; updates come via broadcast
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        clients.discard(ws)
        logger.info("Client disconnected (%d remaining)", len(clients))


async def broadcast(payload: dict):
    if not clients:
        return
    msg = json.dumps(payload, default=str)
    dead = set()
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ── ZMQ consumer ─────────────────────────────────────────────────────────────

async def zmq_reader():
    global book_state, latest_diag

    ctx = zmq.asyncio.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")
    sub.connect("tcp://127.0.0.1:5556")
    sub.setsockopt(zmq.SUBSCRIBE, b"")   # subscribe to all topics

    logger.info("ZMQ SUB connected to :5555 and :5556")

    while True:
        try:
            parts = await sub.recv_multipart()
            if len(parts) < 2:
                continue
            topic, raw = parts[0], parts[1]
            payload = json.loads(raw.decode())

            if topic == b"DOLLAR_BAR":
                bar_buffer.append(payload)
                await broadcast({"type": "bar", "bar": payload})

            elif topic == b"BOOK_TICKER":
                book_state = payload
                await broadcast({"type": "book", "book": payload})

            elif topic == b"DIAGNOSTICS":
                diag_buffer.append(payload)
                latest_diag = payload
                await broadcast({"type": "diag", "diag": payload})

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("ZMQ read error: %s", e)
            await asyncio.sleep(1)

    sub.close()
    ctx.term()


@app.on_event("startup")
async def startup():
    asyncio.create_task(zmq_reader())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8050,
        log_level="warning",
    )
