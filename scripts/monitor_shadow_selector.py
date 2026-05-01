#!/usr/bin/env python3
import zmq
import json
import time
import pandas as pd
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from collections import deque
import numpy as np

# Config
ZMQ_ADDR = "tcp://127.0.0.1:5556"
LOG_DIR = Path("logs/shadow_audit")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = LOG_DIR / "shadow_decisions.parquet"

class ShadowMonitor:
    def __init__(self):
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(ZMQ_ADDR)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"DIAGNOSTICS")
        
        self.history = deque(maxlen=1000)
        self.latest = None
        self.start_time = time.time()
        self.counts = {"flat": 0, "v2": 0, "consensus": 0}
        self.reasons = {}

    def update(self):
        try:
            parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            if len(parts) >= 2:
                payload = json.loads(parts[1].decode())
                if payload.get("type") == "shadow_selector":
                    data = payload["data"]
                    self.latest = data
                    self.history.append(data)
                    
                    # Update stats
                    model = data.get("model", "flat")
                    self.counts[model] = self.counts.get(model, 0) + 1
                    reason = data.get("reason", "unknown")
                    self.reasons[reason] = self.reasons.get(reason, 0) + 1
                    
                    # Save to Parquet (Buffered to avoid IO overhead every bar)
                    if len(self.history) % 10 == 0:
                        self.save_history()
                    return True
        except zmq.Again:
            pass
        return False

    def save_history(self):
        df = pd.DataFrame(list(self.history))
        if not PARQUET_PATH.exists():
            df.to_parquet(PARQUET_PATH, index=False)
        else:
            # Append logic (simplified for script)
            existing = pd.read_parquet(PARQUET_PATH)
            combined = pd.concat([existing, df]).drop_duplicates(subset=['ts']).tail(10000)
            combined.to_parquet(PARQUET_PATH, index=False)

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        return layout

    def get_header(self):
        return Panel(
            f"[bold cyan]Institutional Shadow-Live Selector Monitor[/] | Uptime: {int(time.time() - self.start_time)}s | Decisions: {sum(self.counts.values())}",
            box=box.ROUNDED, style="cyan"
        )

    def get_current_state(self):
        if not self.latest:
            return Panel("Waiting for data...", title="Current State")
        
        data = self.latest
        table = Table(show_header=False, box=box.SIMPLE)
        
        model_style = "green" if data['model'] != "flat" else "white"
        table.add_row("Selected Model", f"[{model_style}]{data['model'].upper()}[/]")
        table.add_row("Reason", f"[yellow]{data['reason']}[/]")
        
        def prob_color(p):
            if p > 0.60: return "green"
            if p > 0.52: return "cyan"
            return "white"

        table.add_row("Alpha Prob Final", f"[{prob_color(data['final_alpha'])}]{data['final_alpha']:.4f}[/]")
        table.add_row("Prob v1 (2024)", f"[{prob_color(data['prob_v1'])}]{data['prob_v1']:.4f}[/]")
        table.add_row("Prob v2 (2025)", f"[{prob_color(data['prob_v2'])}]{data['prob_v2']:.4f}[/]")
        
        return Panel(table, title="[bold]Current State[/]", border_style="blue")

    def get_features_panel(self):
        if not self.latest:
            return Panel("Waiting...", title="Regime Features")
            
        data = self.latest
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Feature")
        table.add_column("Value")
        table.add_column("Threshold")
        
        wvf_style = "bold red" if data['wvf_z'] > 1.5 else "white"
        table.add_row("WVF Z-Score", f"[{wvf_style}]{data['wvf_z']:.2f}[/]", "> 1.5")
        
        div_style = "bold green" if data['cvd_div'] == 1 else "white"
        table.add_row("CVD Divergence", f"[{div_style}]{data['cvd_div']}[/]", "== 1")
        
        table.add_row("HMA Slope", f"{data['hma_slope']:.6f}", "-")
        
        return Panel(table, title="[bold]Regime Indicators[/]", border_style="magenta")

    def get_stats_panel(self):
        total = sum(self.counts.values()) or 1
        table = Table(title="Decision Distribution", box=box.SIMPLE)
        table.add_column("Model/Reason")
        table.add_column("Count")
        table.add_column("%")
        
        for k, v in self.counts.items():
            table.add_row(k, str(v), f"{(v/total)*100:.1f}%")
        
        table.add_section()
        for k, v in self.reasons.items():
            table.add_row(f"[dim]{k}[/]", str(v), f"{(v/total)*100:.1f}%")
            
        return Panel(table, border_style="cyan")

def main():
    monitor = ShadowMonitor()
    console = Console()
    
    with Live(monitor.generate_layout(), refresh_per_second=4, screen=True) as live:
        while True:
            monitor.update()
            
            layout = monitor.generate_layout()
            layout["header"].update(monitor.get_header())
            layout["left"].update(monitor.get_current_state())
            layout["right"].split(
                Layout(monitor.get_features_panel()),
                Layout(monitor.get_stats_panel())
            )
            layout["footer"].update(Panel(f"Log: {PARQUET_PATH.absolute()}", style="dim"))
            
            live.update(layout)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
