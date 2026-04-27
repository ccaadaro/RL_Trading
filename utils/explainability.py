"""
utils/explainability.py
───────────────────────
Feature attribution and explainability tools for the RL trading agent.

Two complementary methods:

  1. PermutationImportance  — black-box, model-agnostic.
     For each feature, permute it across the eval sequence and measure the
     change in action distribution.  No access to model internals required.
     Fast enough for 120 features × 2–3k bars in < 60 s.

  2. GradientAttribution   — white-box, PyTorch only.
     Computes input gradients w.r.t. the policy's action logit at each bar.
     Shows which features drove each individual trade decision.
     Works with any SB3 MlpPolicy or MlpLstmPolicy.

  3. generate_report()      — HTML report combining both methods, saved to
     reports/explainability_<timestamp>.html.

Usage
-----
    from utils.explainability import generate_report
    generate_report(model, val_df, feature_cols, cfg, save_dir="reports")
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import datetime


# ─────────────────────────────────────────────────────────────────────────────
# 1. Permutation Importance  (fast, black-box)
# ─────────────────────────────────────────────────────────────────────────────

class PermutationImportance:
    """
    Model-agnostic feature importance via random permutation.

    For each feature f:
      - Run model on original obs sequence → baseline actions
      - Shuffle column f across all steps → permuted actions
      - importance(f) = mean |permuted_action - baseline_action|

    Repeats n_repeats times and reports mean ± std.

    Parameters
    ----------
    model : any SB3 model with .predict(obs, state, episode_start).
    n_repeats : number of shuffles per feature (3–5 is usually sufficient).
    random_state : numpy random seed for reproducibility.
    """

    def __init__(self, model, n_repeats: int = 3, random_state: int = 42):
        self.model        = model
        self.n_repeats    = n_repeats
        self.rng          = np.random.RandomState(random_state)

    def compute(
        self,
        obs_array: np.ndarray,
        feature_names: List[str],
        max_steps: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        obs_array : shape (n_steps, n_features) — normalised observations.
        feature_names : column labels matching obs_array columns.
        max_steps : if > 0, subsample to save time.

        Returns
        -------
        DataFrame with columns: feature, importance_mean, importance_std,
        sorted descending by importance_mean.
        """
        if max_steps > 0 and len(obs_array) > max_steps:
            idx = self.rng.choice(len(obs_array), max_steps, replace=False)
            idx.sort()
            obs_array = obs_array[idx]

        n_steps, n_feat = obs_array.shape
        assert len(feature_names) == n_feat, \
            f"feature_names length {len(feature_names)} ≠ obs_array columns {n_feat}"

        baseline = self._run_model(obs_array)           # shape (n_steps,)
        importances = np.zeros((n_feat, self.n_repeats))

        for f in range(n_feat):
            for r in range(self.n_repeats):
                shuffled        = obs_array.copy()
                shuffled[:, f]  = self.rng.permutation(shuffled[:, f])
                permuted        = self._run_model(shuffled)
                importances[f, r] = np.mean(np.abs(permuted - baseline))

        return (
            pd.DataFrame({
                "feature":          feature_names,
                "importance_mean":  importances.mean(axis=1),
                "importance_std":   importances.std(axis=1),
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    def _run_model(self, obs_array: np.ndarray) -> np.ndarray:
        """Run model step-by-step, maintaining LSTM state; return action array."""
        actions      = np.empty(len(obs_array), dtype=float)
        lstm_state   = None
        ep_start     = np.ones((1,), dtype=bool)

        for t, obs in enumerate(obs_array):
            raw, lstm_state = self.model.predict(
                obs.reshape(1, -1),
                state=lstm_state,
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start   = np.zeros((1,), dtype=bool)
            actions[t] = float(np.asarray(raw).flat[0])

        return actions


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient Attribution  (white-box, per-trade)
# ─────────────────────────────────────────────────────────────────────────────

class GradientAttribution:
    """
    Input-gradient attribution for SB3 MlpPolicy / MlpLstmPolicy.

    For each bar flagged as a trade entry (action switches to LONG), computes
    ∂action_mean/∂obs via backprop through the policy's actor network.
    The absolute gradient magnitude is used as the attribution score.

    Requires torch.  Falls back gracefully if unavailable.
    """

    def __init__(self, model):
        self.model  = model
        self.policy = model.policy

    def attribute_trades(
        self,
        obs_array: np.ndarray,
        actions: np.ndarray,
        feature_names: List[str],
        top_k: int = 10,
        max_trades: int = 50,
    ) -> List[Dict]:
        """
        For each LONG entry in `actions`, compute gradient attribution.

        Returns
        -------
        List of dicts, one per trade:
          {bar_idx, top_features: DataFrame(feature, attribution, obs_value)}
        """
        try:
            import torch
        except ImportError:
            warnings.warn("torch not available — gradient attribution skipped")
            return []

        device = next(self.policy.parameters()).device
        results = []

        # Identify bars where the agent goes LONG (action switches from 0 → 1)
        entry_bars = [
            i for i in range(1, len(actions))
            if actions[i] >= 0.5 and actions[i - 1] < 0.5
        ][:max_trades]

        for idx in entry_bars:
            obs_t = torch.FloatTensor(obs_array[idx]).unsqueeze(0).to(device)
            obs_t.requires_grad_(True)

            try:
                with torch.enable_grad():
                    # Extract features → latent_pi → action distribution mean
                    feats      = self.policy.extract_features(
                        obs_t, self.policy.pi_features_extractor
                    )
                    latent_pi, _ = self.policy.mlp_extractor(feats)
                    dist         = self.policy._get_action_dist_from_latent(latent_pi)

                    # For Gaussian policy: mean is the deterministic action
                    if hasattr(dist, "distribution"):
                        score = dist.distribution.mean.sum()
                    else:
                        score = dist.mode().sum()

                    score.backward()

                grad = obs_t.grad.detach().cpu().numpy().flatten()
                attr = np.abs(grad)
                top_idx = np.argsort(attr)[::-1][:top_k]

                results.append({
                    "bar_idx": idx,
                    "top_features": pd.DataFrame({
                        "feature":     [feature_names[j] for j in top_idx],
                        "attribution": attr[top_idx],
                        "obs_value":   obs_array[idx, top_idx],
                    }),
                })
            except Exception as e:
                warnings.warn(f"Gradient attribution failed at bar {idx}: {e}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML Report
# ─────────────────────────────────────────────────────────────────────────────

def _make_importance_fig(importance_df: pd.DataFrame, n_top: int = 25) -> str:
    """Return a plotly bar chart as an HTML div string."""
    try:
        import plotly.graph_objects as go
        top = importance_df.head(n_top)
        fig = go.Figure(go.Bar(
            x=top["importance_mean"][::-1],
            y=top["feature"][::-1],
            error_x=dict(type="data", array=top["importance_std"][::-1].tolist()),
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title="Feature Importance (Permutation)",
            xaxis_title="Mean action change when feature is shuffled",
            height=max(400, n_top * 22),
            margin=dict(l=250, r=30, t=50, b=40),
            template="plotly_white",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        return "<p>plotly not installed — install with: pip install plotly</p>"


def _make_trade_table(trade_attrs: List[Dict], n_trades: int = 10) -> str:
    """Return an HTML table of top attributed features for the first N trades."""
    if not trade_attrs:
        return "<p>No gradient attributions available.</p>"

    rows = []
    for t in trade_attrs[:n_trades]:
        feat_strs = ", ".join(
            f"{r['feature']} ({r['attribution']:.4f})"
            for _, r in t["top_features"].head(5).iterrows()
        )
        rows.append(f"<tr><td>Bar {t['bar_idx']}</td><td>{feat_strs}</td></tr>")

    return (
        "<table border='1' style='border-collapse:collapse;font-size:12px'>"
        "<tr><th>Trade entry bar</th><th>Top-5 driving features</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def generate_report(
    model,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Dict,
    save_dir: str = "reports",
    n_perm_repeats: int = 3,
    max_steps: int = 2000,
    turbulence_col: str = "turbulence_feature",
) -> Path:
    """
    Generate a full explainability HTML report.

    Steps
    -----
    1. Build observation array from eval_df (feature columns only, scaled as-is).
    2. Run the model on the eval set, collecting actions.
    3. Compute PermutationImportance over the obs array.
    4. Compute GradientAttribution for LONG trade entries.
    5. Write report to save_dir/explainability_<timestamp>.html.

    Parameters
    ----------
    model : trained RecurrentPPO model.
    eval_df : DataFrame for the evaluation period (must contain feature_cols).
    feature_cols : list of observation column names.
    cfg : CONFIG dict from train2.py (for env building during eval run).
    save_dir : directory to write the HTML report.
    n_perm_repeats : permutation repeats (3 is fast; 5 is more stable).
    max_steps : subsample obs for permutation importance if df is long.
    turbulence_col : column in eval_df containing normalised turbulence.

    Returns
    -------
    Path to the written HTML file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = save_path / f"explainability_{ts}.html"

    print(f"\n[Explainability] Starting report generation...")
    print(f"  Eval period : {eval_df.index[0].date()} → {eval_df.index[-1].date()}")
    print(f"  Features    : {len(feature_cols)}")

    # ── 1. Build obs array ─────────────────────────────────────────────────────
    existing_feat_cols = [c for c in feature_cols if c in eval_df.columns]
    obs_array = eval_df[existing_feat_cols].fillna(0.0).values.astype(np.float32)
    n_steps, n_feat = obs_array.shape
    print(f"  Obs array   : {n_steps} steps × {n_feat} features")

    # ── 2. Collect actions from a single eval run ───────────────────────────────
    print("  Running model for baseline actions...")
    perm_imp   = PermutationImportance(model, n_repeats=n_perm_repeats)
    actions    = perm_imp._run_model(obs_array)      # reuse internal method

    pct_long = (actions >= 0.5).mean() * 100
    n_trades = int(np.sum(np.abs(np.diff(actions >= 0.5))))
    print(f"  Actions     : {pct_long:.1f}% LONG  |  {n_trades} position changes")

    # ── 3. Permutation importance ───────────────────────────────────────────────
    print(f"  Computing permutation importance ({n_perm_repeats} repeats × {n_feat} features)...")
    importance_df = perm_imp.compute(obs_array, existing_feat_cols, max_steps=max_steps)
    top5 = importance_df.head(5)["feature"].tolist()
    print(f"  Top-5 features: {top5}")

    # ── 4. Gradient attribution ─────────────────────────────────────────────────
    print("  Computing gradient attributions for trade entries...")
    grad_attr  = GradientAttribution(model)
    trade_attrs = grad_attr.attribute_trades(
        obs_array, actions, existing_feat_cols, top_k=10
    )
    print(f"  Attributed {len(trade_attrs)} trade entries")

    # ── 5. Build turbulence stats ──────────────────────────────────────────────
    turb_series = eval_df[turbulence_col].fillna(0) if turbulence_col in eval_df.columns \
                  else pd.Series(0, index=eval_df.index)
    turb_mean   = float(turb_series.mean())
    turb_pct95  = float(turb_series.quantile(0.95))
    turb_cash   = float((turb_series >= 2.0).mean() * 100)

    # ── 6. Assemble HTML ───────────────────────────────────────────────────────
    importance_chart = _make_importance_fig(importance_df)
    trade_table      = _make_trade_table(trade_attrs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RL Agent Explainability Report — {ts}</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1    {{ color: #1a3a5c; }}
    h2    {{ color: #2c5f8a; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th,td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th    {{ background: #1a3a5c; color: white; }}
    tr:nth-child(even) {{ background: #f5f9ff; }}
    .stat {{ display:inline-block; margin:8px 16px 8px 0;
             background:#e8f0f8; padding:8px 14px; border-radius:4px; }}
  </style>
</head>
<body>
<h1>RL Agent Explainability Report</h1>
<p><b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}&nbsp;&nbsp;
   <b>Period:</b> {eval_df.index[0].date()} → {eval_df.index[-1].date()}</p>

<h2>Summary</h2>
<span class="stat"><b>Steps:</b> {n_steps}</span>
<span class="stat"><b>Features:</b> {n_feat}</span>
<span class="stat"><b>% LONG:</b> {pct_long:.1f}%</span>
<span class="stat"><b>Position changes:</b> {n_trades}</span>
<span class="stat"><b>Turbulence mean:</b> {turb_mean:.3f}</span>
<span class="stat"><b>Turbulence p95:</b> {turb_pct95:.3f}</span>
<span class="stat"><b>% risk-off (≥2.0):</b> {turb_cash:.1f}%</span>

<h2>Feature Importance (Permutation)</h2>
<p>Mean absolute change in agent action when each feature is randomly shuffled.
   Higher = the agent relies more on this feature.</p>
{importance_chart}

<h2>Top 20 Features (Table)</h2>
{importance_df.head(20).to_html(index=False, float_format="{:.6f}".format)}

<h2>Trade-level Attribution (Gradient)</h2>
<p>For each LONG entry bar, the features with the largest gradient magnitude
   in the policy's actor network output — the model's "reasons" for going long.</p>
{trade_table}

<h2>Notes</h2>
<ul>
  <li>Permutation importance is model-agnostic and handles LSTM state correctly
      (full sequential pass per feature shuffle).</li>
  <li>Gradient attribution uses ∂action_mean/∂obs at the entry bar only
      (no LSTM cross-step gradient — single-step approximation).</li>
  <li>Turbulence ≥ 2.0 triggers the MultiLevelRiskWrapper cash override —
      those bars are excluded from trading regardless of agent output.</li>
</ul>
</body>
</html>"""

    out_file.write_text(html, encoding="utf-8")
    print(f"\n  Report saved → {out_file}")
    return out_file
