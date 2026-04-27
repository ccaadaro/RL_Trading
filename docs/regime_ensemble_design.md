# Regime-Switching Ensemble — Production Design

## Problem

Empirically (iter1/iter2), PPO and SAC converge to different specialisations
on the same env + reward + data:

| | VAL α (bull) | TEST α (bear) |
|---|---|---|
| PPO iter2 | **−41%** | +22% |
| SAC @150k | −75% | **+40%** |

Both signals are useful — PPO is less wrong in bull, SAC captures most of
the short-side alpha in bear. The question is whether we can **route**
between them at inference time based on the market regime.

## Gating signal

We already compute two regime-relevant features per bar:

- `turbulence_feature` — rolling Mahalanobis distance over returns
  (Kritzman 2010). > 1.0 = 90th percentile, > 1.5 = crisis.
- `trend_return_180_feature` — log return over 180 bars (7.5 days).

Simple, interpretable rule:

```
if trend_return_180 > +0.05:      use bull_model   (PPO iter3)
elif trend_return_180 < -0.05:    use bear_model   (SAC)
else:                             use bull_model   (safer default)
if turbulence > 1.5:              force_flat       (risk override)
```

Thresholds to tune post-training. Start conservative — the cost of a
wrong gate is lower than the cost of flipping between policies every bar.

## Gate hysteresis

Hysteresis is **mandatory** in production. Without it, the gate will
flicker near the trend_return threshold and force position flips →
garbage trades. Use two thresholds per transition:

```
bull → bear  requires trend_return_180 < -0.08
bear → bull  requires trend_return_180 > +0.03
```

Hold last regime when inside the band.

## State across switches

Each RL policy keeps no hidden state (MLP, not LSTM). But the
environment does: `bars_in_trade`, drawdown peak, position.
Pragmatic rule: **no position transfer across switches**. When gate
flips, force flat for one bar, then let the new specialist take over.
This avoids "SAC inherits a PPO long in bear regime" pathologies.

## Implementation shape

```python
class RegimeEnsemble:
    def __init__(self, bull_model, bear_model,
                 t_bull_up=0.03, t_bear_down=-0.08, turb_cash=1.5):
        self.bull, self.bear = bull_model, bear_model
        self._state = "bull"   # default
        ...

    def predict(self, obs, *, trend_180, turbulence):
        if turbulence >= self.turb_cash:
            return 0.0  # force flat

        # Hysteresis
        if self._state == "bull" and trend_180 < self.t_bear_down:
            self._state = "bear"
            return 0.0  # transition bar = flat
        if self._state == "bear" and trend_180 > self.t_bull_up:
            self._state = "bull"
            return 0.0

        model = self.bull if self._state == "bull" else self.bear
        action, _ = model.predict(obs, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])
```

Works across PPO + SAC because SB3 `predict()` is uniform.
VecNormalize stats from each specialist must be loaded independently.

## Training budget

- Bull specialist: train on full data with `VAL α` selection (our current
  iter3). VAL is bull so this biases toward bull performance.
- Bear specialist: train on full data but **select best by** α on a
  bear-only slice (e.g. 2022-05 → 2022-12 from training data). This
  carves a bear-regime OOF score without leaking TEST.
- After both are trained, backtest the ensemble end-to-end on TEST +
  a held-out chunk to tune the gate thresholds.

## Risk — and why this is worth shipping

**Risk 1**: Regime lag. `trend_return_180` over 7.5 days means the gate
flips ~a week late. Mitigation: use `turbulence` (reacts in hours) as
emergency override and hysteresis wide enough to avoid churn.

**Risk 2**: Overfitting to regime definitions. The 2022 bear may not
look like the next one. Mitigation: diversify the gate signal — e.g.
combine `trend_return_180` with `volatility_180` z-score, or use the
Hamilton regime indicator.

**Risk 3**: Transition costs. Hysteresis + forced-flat at transition
bar helps but still costs ~1 round-trip per regime flip. If the
ensemble flips 4× per year, that's ~20 bps drag — acceptable.

**Why ship anyway**: For a sellable product, interpretability wins.
"Bull Specialist + Bear Specialist + Crisis Override" is a narrative
clients understand and auditors can review. A single monolithic policy
is a black box that nobody trusts after it blows up once.
