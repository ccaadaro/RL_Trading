import zmq
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShadowMonitor")

# --- Configuration ---
ZMQ_DIAG_ADDR = "tcp://127.0.0.1:5556"
DEPLOY_DIR = Path("/home/nosferatu/freqtrade/user_data/strategies/RL_Trading/deployments/baseline_hardened_v1")
REPLAY_DATA = DEPLOY_DIR / "replay_data.feather"

class ShadowMonitor:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(ZMQ_DIAG_ADDR)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Load Baseline OOF Stats
        logger.info(f"Loading OOF baseline from {REPLAY_DATA}...")
        try:
            # Note: requires pandas/pyarrow in the running environment
            df_oof = pd.read_feather(REPLAY_DATA)
            self.baseline = {
                "meta_p50": df_oof["meta_prob"].median(),
                "meta_std": df_oof["meta_prob"].std(),
                "spread_p90": df_oof["spread_bps"].quantile(0.9),
                "entry_rate": (df_oof["target_pos"] > 0).mean(),
                "veto_rate": (df_oof["meta_prob"] < 0.60).mean()
            }
            logger.info(f"Baseline loaded: {self.baseline}")
        except Exception as e:
            logger.warning(f"Could not load baseline data: {e}. Using heuristics.")
            self.baseline = {
                "meta_p50": 0.35, "meta_std": 0.15, "spread_p90": 8.0,
                "entry_rate": 0.05, "veto_rate": 0.85
            }

        self.live_data = {
            "meta_probs": [],
            "alpha_probs": [],
            "spreads": [],
            "entries": 0,
            "total": 0
        }

    def run(self):
        logger.info(f"Monitoring Shadow-Live on {ZMQ_DIAG_ADDR}...")
        while True:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg)
                self._process_message(data)
            except zmq.Again:
                time.sleep(1)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)

    def _process_message(self, data):
        self.live_data["total"] += 1
        m_prob = data.get("meta_prob", 0.0)
        s_bps = data.get("spread_bps", 0.0)
        t_pos = data.get("target_pos", 0.0)
        
        self.live_data["meta_probs"].append(m_prob)
        self.live_data["spreads"].append(s_bps)
        if t_pos > 0: self.live_data["entries"] += 1
        
        # Every 100 updates, perform health check
        if self.live_data["total"] % 100 == 0:
            self._check_health()

    def _check_health(self):
        l = self.live_data
        b = self.baseline
        
        meta_p50_live = np.median(l["meta_probs"])
        spread_p90_live = np.percentile(l["spreads"], 90)
        entry_rate_live = l["entries"] / l["total"]
        veto_rate_live = sum(1 for p in l["meta_probs"] if p < 0.60) / len(l["meta_probs"])
        
        logger.info(f"--- Health Check #{l['total'] // 100} ---")
        
        # Alert 1: Meta-Prob Drift
        drift = abs(meta_p50_live - b["meta_p50"])
        if drift > 2 * b["meta_std"]:
            logger.warning(f"ALERT: Meta-prob drift detected! Live P50={meta_p50_live:.3f} vs OOF P50={b['meta_p50']:.3f}")

        # Alert 2: Toxic Spreads
        if spread_p90_live > b["spread_p90"] * 1.5:
            logger.warning(f"ALERT: Toxic spreads! Live P90={spread_p90_live:.1f} vs OOF P90={b['spread_p90']:.1f}")

        # Alert 3: Entry Frequency
        if entry_rate_live > 2 * b["entry_rate"]:
            logger.warning(f"ALERT: Abnormal entry frequency! Live={entry_rate_live:.2%} vs Replay={b['entry_rate']:.2%}")

        # Alert 4: Veto Malfunction
        if veto_rate_live < 0.50 or veto_rate_live > 0.99:
            logger.warning(f"ALERT: Abnormal veto rate! Live={veto_rate_live:.2%} (Possible model failure)")

        # Clear window
        l["meta_probs"] = []
        l["spreads"] = []
        l["entries"] = 0
        l["total"] = 0

if __name__ == "__main__":
    monitor = ShadowMonitor()
    monitor.run()
