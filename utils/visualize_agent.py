#!/usr/bin/env python3
"""Interactive agent behaviour visualiser.

Reads a checkpoint trade-log CSV (saved during training) plus the price data
cache and produces a self-contained HTML dashboard you can open in any browser.

Usage
-----
# Latest checkpoint (auto-detect)
python utils/visualize_agent.py

# Specific checkpoint
python utils/visualize_agent.py --trades folds/trades/trades_800000.csv

# Compare two checkpoints
python utils/visualize_agent.py --trades folds/trades/trades_800000.csv folds/trades/trades_1600000.csv

# Show features the agent sees at each trade
python utils/visualize_agent.py --trades folds/trades/trades_800000.csv --features

Output: reports/agent_report_<step>.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR   = Path(__file__).resolve().parent.parent
TRADES_DIR = BASE_DIR / "folds" / "trades"
REPORT_DIR = BASE_DIR / "reports"
CACHE_DIR  = BASE_DIR / "cache"
DATA_ROOT  = BASE_DIR.parent.parent.parent / "data"


# ─────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────

def load_price_data() -> pd.DataFrame:
    """Load BTC 4h price from the parquet cache (fastest) or raw feather."""
    caches = sorted(CACHE_DIR.glob("data_v1_*.parquet"), key=lambda p: p.stat().st_mtime)
    if caches:
        df = pd.read_parquet(caches[-1], columns=["close", "open", "high", "low", "volume"])
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    # Fallback: raw 1h spot data resampled to 4h
    spot = DATA_ROOT / "binance" / "BTC_USDT-1h.feather"
    if spot.exists():
        df = pd.read_feather(spot)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return df

    raise FileNotFoundError("No price data found. Run training once to build the cache.")


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_feature_data() -> pd.DataFrame | None:
    """Load full feature matrix from cache (may be large)."""
    caches = sorted(CACHE_DIR.glob("data_v1_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not caches:
        return None
    try:
        df = pd.read_parquet(caches[-1])
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def compute_portfolio_series(trades: pd.DataFrame, price: pd.DataFrame,
                              val_start: str | None = None,
                              val_end:   str | None = None) -> pd.DataFrame:
    """Reconstruct per-bar portfolio value and compare to buy-and-hold."""
    if val_start:
        price = price[price.index >= pd.Timestamp(val_start)]
    if val_end:
        price = price[price.index <= pd.Timestamp(val_end)]
    if price.empty:
        return pd.DataFrame()

    # Build a position series: step through price bars, apply trades
    pos_map: dict[pd.Timestamp, float] = {}
    current_pos = 0.0
    for _, row in trades.iterrows():
        dt = pd.Timestamp(row["date"])
        current_pos = float(row["position_after"])
        pos_map[dt] = current_pos

    # Forward-fill positions onto the price index
    all_dates = price.index
    positions = pd.Series(0.0, index=all_dates)
    for dt, pos in sorted(pos_map.items()):
        mask = positions.index >= dt
        positions[mask] = pos

    # Portfolio returns
    log_ret = np.log(price["close"] / price["close"].shift(1)).fillna(0.0)
    port_ret = positions.shift(1).fillna(0.0) * log_ret

    port_value = np.exp(port_ret.cumsum()) * 100
    bh_value   = np.exp(log_ret.cumsum()) * 100

    result = pd.DataFrame({
        "date":       all_dates,
        "price":      price["close"].values,
        "portfolio":  port_value.values,
        "bh":         bh_value.values,
        "position":   positions.values,
        "log_return": log_ret.values,
    })
    result = result.set_index("date")

    # Drawdown
    roll_max = result["portfolio"].cummax()
    result["drawdown"] = (result["portfolio"] - roll_max) / roll_max * 100

    return result


def compute_trade_pnl(trades: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Pair up BUY/SELL events and compute per-trade P&L."""
    records = []
    open_trade = None
    for _, row in trades.iterrows():
        if row["action"] in ("BUY", "LONG"):
            open_trade = row
        elif row["action"] in ("SELL", "SHORT", "CLOSE") and open_trade is not None:
            entry_price = float(open_trade["price"])
            exit_price  = float(row["price"])
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            duration_bars = int(row["step"]) - int(open_trade["step"])
            records.append({
                "entry_date":  open_trade["date"],
                "exit_date":   row["date"],
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "pnl_pct":     pnl_pct,
                "duration_bars": duration_bars,
                "win":         pnl_pct > 0,
            })
            open_trade = None
    return pd.DataFrame(records) if records else pd.DataFrame()


def sharpe(returns: pd.Series, periods_per_year: int = 6 * 24 * 365 // 4) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


# ─────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────

def _position_color(pos: float) -> str:
    if pos > 0:
        return "rgba(0,180,0,0.15)"
    if pos < 0:
        return "rgba(220,0,0,0.15)"
    return "rgba(128,128,128,0.05)"


def build_price_position_chart(port: pd.DataFrame, trades: pd.DataFrame,
                                step_label: str) -> go.Figure:
    """Candlestick / line price chart with position background and trade markers."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(
            f"BTC Price  |  Agent Positions  ({step_label})",
            "Portfolio vs Buy-and-Hold (rebased = 100)",
            "Drawdown %",
        ),
    )

    dates = port.index

    # ── Price line ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=port["price"],
        line=dict(color="#a0a0ff", width=1),
        name="BTC Price",
    ), row=1, col=1)

    # ── Position background ──────────────────────────────────
    prev_pos = None
    seg_start = None
    for i, (dt, row) in enumerate(port.iterrows()):
        pos = row["position"]
        if pos != prev_pos:
            if prev_pos is not None and seg_start is not None:
                fig.add_vrect(
                    x0=seg_start, x1=dt,
                    fillcolor=_position_color(prev_pos),
                    layer="below", line_width=0,
                    row=1, col=1,
                )
            seg_start = dt
            prev_pos = pos
    if prev_pos is not None and seg_start is not None:
        fig.add_vrect(
            x0=seg_start, x1=dates[-1],
            fillcolor=_position_color(prev_pos),
            layer="below", line_width=0,
            row=1, col=1,
        )

    # ── Trade markers ────────────────────────────────────────
    buys  = trades[trades["action"].isin(["BUY", "LONG"])]
    sells = trades[trades["action"].isin(["SELL", "SHORT", "CLOSE"])]

    fig.add_trace(go.Scatter(
        x=buys["date"], y=buys["price"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=10, color="lime", line=dict(color="darkgreen", width=1)),
        name="BUY",
        hovertemplate="BUY  $%{y:,.0f}<br>%{x}",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sells["date"], y=sells["price"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="red", line=dict(color="darkred", width=1)),
        name="SELL",
        hovertemplate="SELL  $%{y:,.0f}<br>%{x}",
    ), row=1, col=1)

    # ── Portfolio vs BH ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=port["portfolio"],
        line=dict(color="#00d4ff", width=2),
        name="Agent",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=port["bh"],
        line=dict(color="orange", width=1.5, dash="dot"),
        name="Buy & Hold",
    ), row=2, col=1)

    # ── Drawdown ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=port["drawdown"],
        fill="tozeroy",
        fillcolor="rgba(220,0,0,0.25)",
        line=dict(color="rgba(220,0,0,0.6)", width=1),
        name="Drawdown %",
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def build_trade_analysis_chart(trade_pnl: pd.DataFrame) -> go.Figure:
    if trade_pnl.empty:
        return go.Figure().add_annotation(text="No closed trades found", showarrow=False)

    wins   = trade_pnl[trade_pnl["win"]]
    losses = trade_pnl[~trade_pnl["win"]]
    win_rate = len(wins) / len(trade_pnl) * 100
    avg_win  = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Trade P&L distribution  (win rate {win_rate:.1f}%)",
            "P&L % vs Hold duration (bars)",
        ),
    )

    fig.add_trace(go.Histogram(
        x=trade_pnl["pnl_pct"],
        nbinsx=40,
        marker_color=["lime" if w else "red" for w in trade_pnl["win"]],
        name="P&L %",
        hovertemplate="%{x:.2f}%  count=%{y}",
    ), row=1, col=1)
    fig.add_vline(x=0, line_color="white", line_dash="dash", row=1, col=1)
    fig.add_vline(x=avg_win,  line_color="lime", line_dash="dot",
                  annotation_text=f"avg win {avg_win:.2f}%", row=1, col=1)
    fig.add_vline(x=avg_loss, line_color="red",  line_dash="dot",
                  annotation_text=f"avg loss {avg_loss:.2f}%", row=1, col=1)

    colors = ["lime" if w else "red" for w in trade_pnl["win"]]
    fig.add_trace(go.Scatter(
        x=trade_pnl["duration_bars"],
        y=trade_pnl["pnl_pct"],
        mode="markers",
        marker=dict(color=colors, size=6, opacity=0.7),
        name="trades",
        hovertemplate="duration %{x} bars<br>P&L %{y:.2f}%",
    ), row=1, col=2)
    fig.add_hline(y=0, line_color="white", line_dash="dash", row=1, col=2)

    fig.update_layout(template="plotly_dark", height=400, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=40))
    return fig


def build_metrics_table(port: pd.DataFrame, trade_pnl: pd.DataFrame,
                         step_label: str) -> go.Figure:
    ret_series = port["portfolio"].pct_change().dropna()
    total_return   = (port["portfolio"].iloc[-1] / port["portfolio"].iloc[0] - 1) * 100
    bh_return      = (port["bh"].iloc[-1] / port["bh"].iloc[0] - 1) * 100
    max_dd         = port["drawdown"].min()
    sh             = sharpe(np.log(port["portfolio"] / port["portfolio"].shift(1)).dropna())
    n_trades       = len(trade_pnl)
    win_rate       = f"{len(trade_pnl[trade_pnl['win']])/max(n_trades,1)*100:.1f}%" if n_trades else "—"
    avg_dur        = f"{trade_pnl['duration_bars'].mean():.1f} bars" if n_trades else "—"

    metrics = {
        "Checkpoint":      step_label,
        "Total return":    f"{total_return:+.2f}%",
        "Buy & Hold":      f"{bh_return:+.2f}%",
        "Alpha":           f"{total_return - bh_return:+.2f}%",
        "Sharpe":          f"{sh:.3f}",
        "Max drawdown":    f"{max_dd:.2f}%",
        "Trades":          str(n_trades),
        "Win rate":        win_rate,
        "Avg duration":    avg_dur,
    }

    fig = go.Figure(go.Table(
        header=dict(values=["Metric", "Value"],
                    fill_color="#1f1f2e", font=dict(color="white", size=13),
                    align="left"),
        cells=dict(
            values=[list(metrics.keys()), list(metrics.values())],
            fill_color=[["#1a1a2e"] * len(metrics), ["#16213e"] * len(metrics)],
            font=dict(color="white", size=12),
            align="left",
        ),
    ))
    fig.update_layout(template="plotly_dark", height=350,
                      margin=dict(l=20, r=20, t=30, b=10))
    return fig


def build_feature_snapshot(features_df: pd.DataFrame,
                            trades: pd.DataFrame,
                            n_top: int = 20) -> go.Figure:
    """Heatmap of the top-variance features at each BUY/SELL step."""
    trade_dates = pd.to_datetime(trades["date"]).unique()
    feat_cols = [c for c in features_df.columns if c.endswith("_feature")]
    if not feat_cols:
        return go.Figure().add_annotation(text="No feature columns found", showarrow=False)

    # Select top-N features by variance
    variances = features_df[feat_cols].var().sort_values(ascending=False)
    top_cols = variances.head(n_top).index.tolist()

    # Sample features at trade timestamps
    idx = features_df.index.get_indexer(trade_dates, method="nearest")
    subset = features_df.iloc[idx][top_cols]
    subset.index = trade_dates

    # Normalise each column to [-1, 1] for display
    normed = (subset - subset.mean()) / (subset.std() + 1e-9)
    normed = normed.clip(-3, 3)

    # Label rows with action
    action_map = dict(zip(pd.to_datetime(trades["date"]), trades["action"]))
    row_labels = [f"{action_map.get(d, '?')}  {d.strftime('%Y-%m-%d %H:%M')}"
                  for d in subset.index]

    fig = go.Figure(go.Heatmap(
        z=normed.values,
        x=[c.replace("_feature", "") for c in top_cols],
        y=row_labels,
        colorscale="RdBu",
        zmid=0,
        hovertemplate="Feature: %{x}<br>Trade: %{y}<br>z-score: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"Feature values at each trade (top {n_top} by variance)",
        height=max(400, len(row_labels) * 22),
        margin=dict(l=200, r=20, t=60, b=80),
        xaxis=dict(tickangle=45),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────

def build_report(trades_paths: list[Path], show_features: bool = False) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    price = load_price_data()
    features = load_feature_data() if show_features else None

    html_parts: list[str] = [
        "<html><head><meta charset='utf-8'>"
        "<title>RL Agent Report</title>"
        "<style>body{background:#0a0a1a;color:#e0e0f0;font-family:sans-serif;}"
        "h1,h2{color:#7ab8f5;} .section{margin:20px 0;}</style>"
        "</head><body>",
        "<h1>RL Trading Agent — Behaviour Report</h1>",
    ]

    for trades_path in trades_paths:
        step_label = trades_path.stem.replace("trades_", "step ")
        trades = load_trades(trades_path)

        # Infer val window from trade dates
        if not trades.empty:
            val_start = str(trades["date"].min().date())
            val_end   = str(trades["date"].max().date())
        else:
            val_start = val_end = None

        port      = compute_portfolio_series(trades, price, val_start, val_end)
        trade_pnl = compute_trade_pnl(trades, price)

        html_parts.append(f"<div class='section'><h2>{step_label}</h2>")

        for fig in [
            build_metrics_table(port, trade_pnl, step_label),
            build_price_position_chart(port, trades, step_label),
            build_trade_analysis_chart(trade_pnl),
        ]:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"
                                          if html_parts[0].count("cdn") == 0 else False))

        if show_features and features is not None and not trades.empty:
            html_parts.append(build_feature_snapshot(features, trades).to_html(
                full_html=False, include_plotlyjs=False))

        html_parts.append("</div><hr>")

    html_parts.append("</body></html>")

    # Pick output name from last (best) checkpoint
    out_name = f"agent_report_{trades_paths[-1].stem.replace('trades_', '')}.html"
    out_path = REPORT_DIR / out_name
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"✅ Report saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise RL agent behaviour")
    parser.add_argument("--trades", nargs="+", type=Path,
                        help="Path(s) to trade CSV(s). Default: latest in folds/trades/")
    parser.add_argument("--best", action="store_true",
                        help="Use best checkpoint only (largest Sharpe in log)")
    parser.add_argument("--all",  action="store_true",
                        help="Use all available trade CSVs")
    parser.add_argument("--features", action="store_true",
                        help="Show feature heatmap at each trade (slow)")
    args = parser.parse_args()

    if args.trades:
        paths = [Path(p) for p in args.trades]
    elif args.all:
        paths = sorted(TRADES_DIR.glob("trades_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
    else:
        # Default: latest file
        available = sorted(TRADES_DIR.glob("trades_*.csv"),
                           key=lambda p: int(p.stem.split("_")[1]))
        if not available:
            print("No trade CSV files found in", TRADES_DIR)
            sys.exit(1)
        paths = [available[-1]]

    build_report(paths, show_features=args.features)


if __name__ == "__main__":
    main()
