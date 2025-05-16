# train_rl.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import sys
from stable_baselines3.common.callbacks import BaseCallback
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas_ta as ta
from transformer_extractor import TransformerFeatures
from stable_baselines3.common.callbacks import CallbackList

from trading_env.trading_env import TradingEnv, reward as risk_reward   # ← NUEVO
from stable_baselines3.common.policies import ActorCriticPolicy
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from trading_env.trading_env import TradingEnv
import pandas as pd
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="You provided an OpenAI Gym environment")
from fear_and_greed import FearAndGreedIndex
import numpy as np
from copy import deepcopy


import gymnasium as gym
import numpy as np


def _sharpe_ratio(nav: np.ndarray) -> float:
    rets = np.diff(np.log(nav))
    return (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(8_760)


class ValidationPerformanceCallback(BaseCallback):
    """Detiene entrenamiento si train y validation divergen demasiado"""
    def __init__(self, eval_env, check_freq=10000, max_train_val_ratio=2.0):
        super().__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.max_ratio = max_train_val_ratio
        
    def _evaluate_performance(self, env):
        """Evalúa el rendimiento del modelo en un entorno dado de manera robusta"""
        # Reset del entorno con manejo de diferentes formatos de retorno
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # gym >= 0.26: (obs, info)
        else:
            obs = reset_result
        
        done = False
        valuations = []
        
        # Ejecutar el modelo en deterministic mode con manejo de excepciones
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            
            try:
                # Intenta con VecEnv que devuelve (obs, rewards, dones, infos)
                step_result = env.step(action)
                
                # Manejar diferentes formatos de retorno
                if len(step_result) == 4:  # VecEnv o gym < 0.26
                    next_obs, _, dones, infos = step_result
                    
                    # Verificar si dones es un array o un escalar
                    if isinstance(dones, (list, np.ndarray)):
                        done = dones[0]
                    else:
                        done = dones
                    
                    # Obtener la valoración del portafolio
                    if isinstance(infos, list) and len(infos) > 0:
                        val = infos[0].get("portfolio_valuation", 1.0)
                    else:
                        val = infos.get("portfolio_valuation", 1.0) if isinstance(infos, dict) else 1.0
                    
                elif len(step_result) == 5:  # gym >= 0.26
                    next_obs, _, terminated, truncated, info = step_result
                    done = terminated or truncated
                    
                    if isinstance(info, dict):
                        val = info.get("portfolio_valuation", 1.0)
                    else:
                        val = 1.0
                else:
                    print(f"Formato de retorno no reconocido: {len(step_result)} elementos")
                    return 1.0
                
                valuations.append(val)
                obs = next_obs
                
            except Exception as e:
                print(f"Error durante la evaluación: {e}")
                return 1.0  # Valor por defecto en caso de error
        
        # Calcular rendimiento (retorno total)
        if len(valuations) > 1 and valuations[0] > 0:
            return valuations[-1] / valuations[0]  # Retorno ratio
        else:
            return 1.0  # Sin cambios si no hay suficientes datos
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            train_perf = self._evaluate_performance(self.model.env)
            val_perf = self._evaluate_performance(self.eval_env)
            
            # Si el rendimiento en training es mucho mayor que en validación
            if train_perf > self.max_ratio * val_perf:
                print(f"Early stopping: train_perf={train_perf:.2f} > {self.max_ratio} * val_perf={val_perf:.2f}")
                return False
        return True

class ActionTrackingCallback(BaseCallback):
    """Track action distribution over time"""
    def __init__(self, eval_env, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.eval_env = eval_env
        self.action_counts = [0, 0, 0]  # For [-1, 0, 1]
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Reset counters periodically to see recent trends
            prev_counts = self.action_counts.copy()
            self.action_counts = [0, 0, 0]
            
            # Run a short evaluation to see action distribution
            if self.n_calls > 0:
                obs = self.eval_env.reset()[0]
                for _ in range(100):
                    action, _ = self.model.predict(obs, deterministic=False)
                    
                    # Get the action index for tracking
                    if isinstance(action, np.ndarray) and len(action.shape) > 0:
                        action_idx = int(action[0])
                    else:
                        action_idx = int(action)
                        
                    self.action_counts[action_idx] += 1
                    
                    # Ensure action is properly formatted for VecEnv
                    if not isinstance(action, np.ndarray) or action.shape == ():
                        action_vec = np.array([action_idx])
                    else:
                        action_vec = action  # Already in correct format
                    
                    # FIXED: Proper unpacking for VecEnv step return values (4 values, not 5)
                    obs, rewards, dones, infos = self.eval_env.step(action_vec)
                    done = dones[0]  # Get the done flag for the first (only) environment
                    if done:
                        obs = self.eval_env.reset()[0]
                    
                print(f"\n--- Action Distribution at step {self.n_calls} ---")
                print(f"SHORT: {self.action_counts[0]}%, HOLD: {self.action_counts[1]}%, LONG: {self.action_counts[2]}%")
                
                # Entropy adjustment logic
                if max(self.action_counts) > 90:
                    print("⚠️ ACTION DIVERSITY CRISIS - INCREASING ENTROPY")
                    self.model.ent_coef = max(0.1, float(self.model.ent_coef) * 2)
                elif max(self.action_counts) < 60:
                    if hasattr(self.model, 'ent_coef') and not callable(self.model.ent_coef):
                        current_ent = float(self.model.ent_coef)
                        if current_ent > 0.005:
                            self.model.ent_coef = current_ent * 0.9
        
        return True





class DetailedLoggingCallback(BaseCallback):
    """Logs detailed metrics during training"""
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []
        self.actions = []
        self.values = []
        
    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            # Collect recent data
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                recent_rewards = self.model.rollout_buffer.rewards[-100:].flatten()
                recent_actions = self.model.rollout_buffer.actions[-100:].flatten()
                recent_values = self.model.rollout_buffer.values[-100:].flatten()
                
                self.rewards.extend(recent_rewards)
                self.actions.extend(recent_actions)
                self.values.extend(recent_values)
                
                # Log statistics
                print(f"\n--- Training Stats at step {self.n_calls} ---")
                print(f"Recent reward mean: {np.mean(recent_rewards):.4f}, std: {np.std(recent_rewards):.4f}")
                print(f"Action distribution: {np.bincount(recent_actions.astype(int))}")
                print(f"Value estimate mean: {np.mean(recent_values):.4f}, std: {np.std(recent_values):.4f}")
                
                # Check for issues
                if np.std(recent_actions) < 0.01:
                    print("⚠️ WARNING: Low action diversity - model may be stuck!")
                if np.std(recent_values) < 0.01:
                    print("⚠️ WARNING: Low value diversity - critic may be stuck!")
                
        return True



class PeriodicValidation(BaseCallback):
    """
    Evaluates the policy every `every_ts` timesteps and saves the best model based on the specified metric.
    """
    def __init__(self, 
                 val_env, 
                 every_ts: int = 100_000,
                 save_path: str = "best_model.zip",
                 metric: str = "sharpe",  # Accepted metric parameter
                 verbose: int = 0):
        super().__init__(verbose)
        self.val_env = val_env
        self.every_ts = every_ts
        self.save_path = save_path
        self.metric = metric.lower()
        self.best_val = -np.inf

    @staticmethod
    def _sharpe(nav: np.ndarray) -> float:
        if len(nav) < 2:
            return 0.0
        rets = np.diff(np.log(nav))
        return (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(8_760)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_ts != 0:
            return True

        # Run validation episode
        obs = self.val_env.reset()
        nav = []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.val_env.step(action)
            done = dones[0] or infos[0].get("time_limit", False)
            nav.append(infos[0]["portfolio_valuation"])
        
        # Calculate metric
        if self.metric == "sharpe":
            score = self._sharpe(np.array(nav))
        elif self.metric == "final_return":
            score = nav[-1] / nav[0] - 1
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if self.verbose:
            print(f"[VAL] step={self.num_timesteps:,} {self.metric}: {score:.4f}")

        # Save if improved
        if score > self.best_val:
            self.best_val = score
            self.model.save(self.save_path)
            if self.verbose:
                print(f"  ↳ New best model saved to {self.save_path}")

        return True

    # ------------------------------------------------------------------ #
    def _run_validation(self):
        obs = self.val_env.reset()
        nav = []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.val_env.step(action)
            done = dones[0] or infos[0].get("time_limit", False)
            nav.append(infos[0]["portfolio_valuation"])
        return nav






class EarlyStopKL(BaseCallback):
    """
    Detiene el entrenamiento cuando el KL medio (EMA) permanece por
    debajo de `kl_threshold` durante `patience` rollouts *después* de
    haber superado `min_timesteps` de entrenamiento efectivo.
    """
    def __init__(self,
                 kl_threshold:    float = 5e-4,
                 patience:        int   = 10,
                 warmup_rollouts: int   = 15,
                 min_timesteps:   int   = 200_000,  #  <-- NUEVO
                 ema_gamma:       float = 0.9,
                 verbose: int = 0):
        super().__init__(verbose)
        self.kl_thr        = kl_threshold
        self.patience      = patience
        self.warmup        = warmup_rollouts
        self.min_timesteps = min_timesteps
        self.gamma         = ema_gamma
        self.bad           = 0
        self.ema_kl        = None
        self.rollouts      = 0

    # ——————— eventos de SB3 ————————————————————————————————
    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:                     # a veces no lo loggea
            return

        self.ema_kl = kl if self.ema_kl is None else \
                      self.gamma * self.ema_kl + (1-self.gamma) * kl

        # todavía no vigilamos:
        if (self.rollouts < self.warmup or
            self.num_timesteps < self.min_timesteps):
            return

        if self.ema_kl > self.kl_thr:
            self.bad += 1
            if self.verbose:
                print(f"[KL‑CB] ema_kl={self.ema_kl:.2e}  "
                      f"({self.bad}/{self.patience})")
        else:
            self.bad = 0

    def _on_step(self) -> bool:
        # aborta si se superó la paciencia
        if self.bad >= self.patience:
            if self.verbose:
                print("🟥 Early-stop por KL: sin cambios significativos "
                      f"{self.patience} rollouts seguidos.")
            return False
        return True


class RiskLimit(gym.Wrapper):
    def __init__(self, env, max_leverage=2.0, max_dd=0.25, penalty=-1.0):
        super().__init__(env)
        self.max_leverage = max_leverage
        self.max_dd = max_dd
        self.penalty = penalty
        self.peak_value = None

    # ---------- helpers --------------------------------------------------
    def _leverage(self):
        price = self.env._get_price()
        return abs(self.env._portfolio.real_position(price))

    def _drawdown(self, current_val):
        if self.peak_value is None or self.peak_value == 0:
            return 0.0
        return (self.peak_value - current_val) / self.peak_value

    def _breach(self, info):
        """¿Se supera apalancamiento o DD?"""
        lev = self._leverage()
        dd  = self._drawdown(info["portfolio_valuation"])
        return (lev > self.max_leverage) or (dd > self.max_dd)

    # ---------- Gymnasium interface --------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.peak_value = info["portfolio_valuation"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.peak_value = max(self.peak_value, info["portfolio_valuation"])

        if not terminated and not truncated and self._breach(info):
            # fuerza posición plana
            self.env._trade(0)
            reward += self.penalty
            info["risk_event"] = True
            obs = self.env._get_obs()  # observación tras cerrar
        else:
            info["risk_event"] = False

        return obs, reward, terminated, truncated, info


class AdaptiveRiskLimit(RiskLimit):
    """
    Igual que RiskLimit pero:
      • el apalancamiento máximo se recalcula cada paso en función
        de la volatilidad reciente;
      • la tolerancia de draw‑down aumenta suavemente cuando el bot
        va ganando dinero.
    """
    def __init__(self, env, base_leverage=1.5, volatility_window=72):
        # max_dd y penalty se heredan; puedes ajustarlos aquí si quieres
        super().__init__(env, max_leverage=base_leverage, max_dd=0.25, penalty=-1.0)
        self.base_leverage = base_leverage
        self.vol_window    = volatility_window        # últimas 72 h en un env 1‑h

    # ---------- lógica dinámica -----------------------------------------
    def _dynamic_leverage(self):
        # Retornos logarítmicos de la ventana más reciente
        returns = np.log(self.env.df['close']).diff().tail(self.vol_window)
        vol = returns.std() * np.sqrt(24*365)                # anualizada

        # Regla simple: cuanto más volátil, menor leverage permitido
        return min(self.base_leverage, 2.5 / (vol + 0.05))

    def _dynamic_max_dd(self):
        # Beneficio acumulado bruto desde el inicio del fold
        cur_nav = self.env.historical_info['portfolio_valuation', -1]
        perf    = (cur_nav / self.env.portfolio_initial_value) - 1.0

        # Entre 0.25 (sin beneficio) y 0.50 (+∞ beneficio, asintótico)
        return 0.25 * (1 + np.tanh(perf))
    
    # ---------- sobreescribe step() --------------------------------------
    def step(self, action):
        # Actualiza los límites antes de delegar en el env interior
        self.max_leverage = self._dynamic_leverage()
        self.max_dd       = self._dynamic_max_dd()
        return super().step(action)


class ImprovedRiskLimit(AdaptiveRiskLimit):
    def __init__(self, env, base_leverage=1.3, volatility_window=96):
        super().__init__(env, base_leverage, volatility_window)
        
    def _dynamic_leverage(self):
        # Análisis más detallado de volatilidad con mayor peso a datos recientes
        returns = np.log(self.env.df['close']).diff().tail(self.vol_window)
        recent_vol = returns.tail(24).std() * np.sqrt(24*365)
        overall_vol = returns.std() * np.sqrt(24*365)
        
        # Pondera más la volatilidad reciente
        vol = 0.7 * recent_vol + 0.3 * overall_vol
        
        # Reduce leverage más agresivamente cuando hay alta volatilidad
        if vol > 0.6:  # Volatilidad extrema
            return self.base_leverage * 0.5
        return min(self.base_leverage, 2.0 / (vol + 0.05))
    
    def _dynamic_max_dd(self):
        # Más conservador con el DD permitido durante drawdowns
        cur_dd = self._drawdown(self.env.historical_info['portfolio_valuation', -1])
        cur_nav = self.env.historical_info['portfolio_valuation', -1]
        perf = (cur_nav / self.env.portfolio_initial_value) - 1.0
        
        # Si ya estamos en DD, ser más conservador
        if cur_dd > 0.1:
            return 0.15 * (1 + np.tanh(perf))
        return 0.25 * (1 + np.tanh(perf))


class EntropyDecay(BaseCallback):
    """
    Actualiza model.ent_coef según la función `schedule(progress_rem)`.
    """
    def __init__(self, schedule, total_ts, verbose: int = 0):
        super().__init__(verbose)
        self.schedule  = schedule
        self.total_ts  = total_ts

    # ― se llama tras cada rollout ―
    def _on_rollout_end(self) -> None:
        prog_rem = 1.0 - self.num_timesteps / self.total_ts   # 1→0
        # Asegurarse de que self.model.ent_coef se establece como un float
        current_ent_coef_val = float(self.schedule(prog_rem))
        self.model.ent_coef = current_ent_coef_val

    # ― obligatorio: decide si continuamos el entrenamiento ―
    def _on_step(self) -> bool:
        return True



class ConservativeRiskLimit(ImprovedRiskLimit):
    def __init__(self, env, base_leverage=1.0, volatility_window=144):
        super().__init__(env, base_leverage, volatility_window)
        
    def _dynamic_leverage(self):
        # Análisis más sofisticado de volatilidad
        returns = np.log(self.env.df['close']).diff().tail(self.vol_window)
        volatility_trend = returns.rolling(24).std().diff(24).iloc[-1]
        
        # Volatilidad reciente con más peso
        recent_vol = returns.tail(48).std() * np.sqrt(24*365)
        overall_vol = returns.std() * np.sqrt(24*365)
        combined_vol = 0.8 * recent_vol + 0.2 * overall_vol
        
        # Si la volatilidad está aumentando, reducir más agresivamente
        vol_factor = 1.2 if volatility_trend > 0 else 1.0
        
        # Máximo leverage más conservador
        if combined_vol > 0.7:  # Volatilidad extrema
            return self.base_leverage * 0.3
        elif combined_vol > 0.4:  # Volatilidad alta
            return self.base_leverage * 0.5 / vol_factor
        return min(self.base_leverage, 1.5 / (combined_vol * vol_factor + 0.1))
    
    def _dynamic_max_dd(self):
        # Limita el drawdown permitido de forma más estricta
        cur_dd = self._drawdown(self.env.historical_info['portfolio_valuation', -1])
        
        # Si ya estamos en DD, ser mucho más conservador
        if cur_dd > 0.15:
            return 0.20  # Limitar pérdidas adicionales
        elif cur_dd > 0.1:
            return 0.15 + 0.05 * (1 - cur_dd/0.15)  # Escala entre 0.15-0.20
        
        # Performance actual vs inicial
        cur_nav = self.env.historical_info['portfolio_valuation', -1]
        perf = (cur_nav / self.env.portfolio_initial_value) - 1.0
        
        # Entre 0.20 (sin ganancia) y máximo 0.30 (con mucha ganancia)
        return 0.20 + 0.10 * np.tanh(perf * 0.5)


def make_walk_forward_splits(df, train_hours=24*180, val_hours=24*30, gap_hours=24*7):
    """Añadir gap entre train y validation para prevenir leakage"""
    i_start = 0
    while True:
        train_end = i_start + train_hours
        val_start = train_end + gap_hours  # Gap temporal importante
        val_end = val_start + val_hours
        if val_end > len(df):
            break
        train_slice = df.iloc[i_start:train_end].copy()
        val_slice = df.iloc[val_start:val_end].copy()
        yield train_slice, val_slice
        i_start += val_hours  # Avanzar por validación completa



# -----------------------------------------
# 1. VELAS 1H
# -----------------------------------------
candles = pd.read_json("../../data/binance/BTC_USDT-1h.json", orient="records")
candles.columns = ["timestamp", "open", "high", "low", "close", "volume"]
candles["date"] = pd.to_datetime(candles["timestamp"], unit="ms")
candles.set_index("date", inplace=True)
candles.drop(columns="timestamp", inplace=True)

# -----------------------------------------
# 2. TRADES tick  -> 1H
# -----------------------------------------
cols = ["timestamp", "side", "price", "amount", "cost"]         # solo útiles
dtypes = {"price":"float32", "amount":"float32", "cost":"float32"}

trades = pd.read_feather("../../data/binance/BTC_USDT-trades.feather",
                         columns=cols)
trades = trades.astype(dtypes)
trades["side"] = trades["side"].astype("category")              # 2 bytes por fila
trades["date"] = pd.to_datetime(trades["timestamp"], unit="ms")
trades.set_index("date", inplace=True)

trades["buy_vol"]   = np.where(trades["side"] == "buy",  trades["amount"], 0.0)
trades["sell_vol"]  = np.where(trades["side"] == "sell", trades["amount"], 0.0)
trades["buy_cost"]  = np.where(trades["side"] == "buy",  trades["cost"],   0.0)
trades["sell_cost"] = np.where(trades["side"] == "sell", trades["cost"],   0.0)

agg = {
    "amount":    "sum",
    "buy_vol":   "sum",
    "sell_vol":  "sum",
    "buy_cost":  "sum",
    "sell_cost": "sum",
    "price":     "mean",
    "side":      "count"
}

trades_1h = (
    trades
    .groupby(pd.Grouper(freq="1h"))
    .agg(agg)
    .rename(columns={"side": "trades_n"})
)
trades_1h["vol_imbalance_feature"] = (
    trades_1h["buy_vol"] - trades_1h["sell_vol"]
) / (trades_1h["buy_vol"] + trades_1h["sell_vol"] + 1e-9)
trades_1h["dollar_delta_feature"] = trades_1h["buy_cost"] - trades_1h["sell_cost"]

# -----------------------------------------
# 3. MERGE velas + trades
# -----------------------------------------
df = candles.join(trades_1h, how="left")

# -----------------------------------------
# 4. FEAR & GREED
# -----------------------------------------
from fear_and_greed import FearAndGreedIndex
fng = FearAndGreedIndex()
fng_df = pd.DataFrame(fng.get_last_n_days(365))
fng_df["date"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s")
fng_df.set_index("date", inplace=True)
fng_df["value"] = fng_df["value"].astype(int)

df["fng_feature"] = fng_df["value"].reindex(df.index, method="ffill").fillna(50)

# -----------------------------------------
# 5. INDICADORES
# -----------------------------------------
import pandas_ta as ta

df["atr_feature"] = ta.atr(df["high"], df["low"], df["close"], length=14)
df["volatility_20_feature"] = df["close"].pct_change().rolling(20, min_periods=1).std()

# Calendario
df["hour_sin"] = np.sin(2*np.pi*df.index.hour/24)
df["hour_cos"] = np.cos(2*np.pi*df.index.hour/24)
df["dow_sin"]  = np.sin(2*np.pi*df.index.dayofweek/7)
df["dow_cos"]  = np.cos(2*np.pi*df.index.dayofweek/7)

# Volumen / flujo
df["obv_feature"]  = ta.obv(df["close"], df["volume"])
df["vwap_feature"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"], length=14)
df["mfi_feature"]  = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
df["cci_feature"]  = ta.cci(df["high"], df["low"], df["close"], length=20)

sto = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
df["stoch_k_feature"] = sto["STOCHk_14_3_3"]
df["stoch_d_feature"] = sto["STOCHd_14_3_3"]

# RSI diario mapeado
daily_rsi = ta.rsi(df["close"].resample("1D").last().ffill(), length=14)
df["daily_rsi_feature"] = daily_rsi.reindex(df.index, method="ffill")

# Precio normalizado (sin escalar aún)
# df["price_norm_feature"] = df["close"] / df["close"].iloc[0]
df["ret_1h_feature"]  = np.log(df["close"]).diff()
df["ret_24h_feature"] = np.log(df["close"]).diff(24)

vol_cols = ["volume", "amount", "buy_vol", "sell_vol",
            "buy_cost", "sell_cost"]

for col in vol_cols:
    dz = df[col].groupby(df.index.date)\
                .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df[f"{col}_z_feature"] = dz.fillna(0)
# Liquidity features

df['spread_ratio'] = (df['high'] - df['low'])/df['volume'].rolling(24).mean()

# Smart money indicators
df['whale_flow'] = np.log1p(df['buy_cost'] - np.log1p(df['sell_cost'])).rolling(12).mean()
# For PPO with all components
ppo_result = ta.ppo(df["close"])
for col in ppo_result.columns:
    df[f"ppo_{col.lower()}_feature"] = ppo_result[col]

# For KST with all components
kst_result = ta.kst(df["close"])
for col in kst_result.columns:
    df[f"kst_{col.lower()}_feature"] = kst_result[col]
# Market regime detection
# 1) Clasificación de tendencia + volatilidad
trend_up = df['close'] > df['close'].rolling(200, min_periods=1).mean()

conditions = [
    trend_up & (df['volatility_20_feature'] > 0.03),   # bull + alta vol
    trend_up & (df['volatility_20_feature'] <= 0.03),  # bull + baja vol
    ~trend_up & (df['volatility_20_feature'] > 0.04),  # bear + alta vol
    ~trend_up & (df['volatility_20_feature'] <= 0.04)  # bear + baja vol
]
choices = ['bull_high_vol', 'bull_low_vol',
           'bear_high_vol', 'bear_low_vol']

# 2) Columna string con el régimen
df['market_regime'] = np.select(conditions, choices, default='neutral')



# 3) Mapear a código numérico (int8) para que History acepte float
regime_codes = {
    'bear_low_vol'  : 0,
    'bear_high_vol' : 1,
    'bull_low_vol'  : 2,
    'bull_high_vol' : 3,
    'neutral'       : -1
}

df['market_regime_feature'] = (
    pd.Categorical(df['market_regime'],
                   categories=regime_codes.keys())
      .rename_categories(regime_codes)
      .astype('int8')
)

df['market_regime_code'] = (
    pd.Categorical(df['market_regime'], categories=regime_codes.keys())
      .rename_categories(regime_codes)        # 0‑3
      .astype('int8')
)

df.drop(columns='market_regime', inplace=True)      # opcional

def add_support_resistance(df, window=20, threshold=0.05):
    highs = df['high'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0)
    lows = df['low'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[len(x)//2] == min(x) else 0)
    
    df['at_resistance_feature'] = highs.rolling(window).sum() / window
    df['at_support_feature'] = lows.rolling(window).sum() / window
    
    close = df['close']
    upper_band = close.rolling(window).max()
    lower_band = close.rolling(window).min()
    
    df['range_position_feature'] = (close - lower_band) / (upper_band - lower_band + 1e-9)
    return df

df = add_support_resistance(df)

df['price_vol_divergence_feature'] = (
    (df['close'].pct_change(5) > 0) & 
    (df['volume'].pct_change(5) < 0)
).astype(int) - (
    (df['close'].pct_change(5) < 0) & 
    (df['volume'].pct_change(5) > 0)
).astype(int)
# -----------------------------------------
# 6. LIMPIEZA FINAL
# -----------------------------------------
df.sort_index(inplace=True)
df.fillna(0, inplace=True)
print(df.columns)

# ── helpers ────────────────────────────────────────────────────────────
def longest_drawdown_duration(nav: np.ndarray) -> int:
    peaks   = np.maximum.accumulate(nav)
    in_dd   = nav < peaks
    dur = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        dur = max(dur, cur)
    return dur

def nav_perf_in_regime(nav: np.ndarray,
                       regimes: np.ndarray,
                       tag_code: int) -> float:
    mask = regimes == tag_code
    if mask.sum() < 2:
        return 0.0
    nav_r = nav[mask]
    return np.log(nav_r[-1] / nav_r[0])

# ── métricas del fold ──────────────────────────────────────────────────
def calculate_metrics(history):
    nav = np.asarray(history['portfolio_valuation', :], float)

    # ── si no hay datos suficientes devolvemos ceros ──────────────────
    if nav.size < 2:
        return {
            'annualized_sharpe': 0.0,
            'win_rate':          0.0,
            'profit_factor':     0.0,
            'max_dd_duration':   0,
            'regime_performance': {'bull': 0.0, 'bear': 0.0},
        }

    returns = np.diff(np.log(nav))

    metrics = {
        'annualized_sharpe': (returns.mean() / (returns.std() + 1e-9))
                             * np.sqrt(8_760),
        'win_rate':          (returns > 0).mean(),
        'profit_factor':     returns[returns > 0].sum()
                             / abs(returns[returns < 0].sum() + 1e-12),
        'max_dd_duration':   longest_drawdown_duration(nav),
        'regime_performance': {
            'bull': nav_perf_in_regime(
                        nav,
                        history['data_market_regime_code', :],
                        tag_code=2   # bull_low_vol
                     ),
            'bear': nav_perf_in_regime(
                        nav,
                        history['data_market_regime_code', :],
                        tag_code=0   # bear_low_vol
                     ),
        },
    }
    return metrics
def cosine_decay_lr(progress_remaining: float,
                    base_lr: float = 5e-5,
                    min_lr:  float = 1e-6,
                    plateau_frac: float = 0.15) -> float:
    """
    • progress_remaining llega con 1 al principio y 0 al final.
    • Durante el último `plateau_frac` mantenemos `min_lr` plano.
    """
    # tramo plano al final
    if progress_remaining < plateau_frac:
        return min_lr

    # mapeamos 1→0 (inicio) a 0→1 para el coseno
    t = 1.0 - (progress_remaining - plateau_frac) / (1.0 - plateau_frac)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))

def cosine_lr(progress, base=3e-5, min_lr=5e-6):
    # progress = 1 → 0
    cos_inner = np.pi * (1.0 - progress)          # 0→π
    return min_lr + 0.5*(base-min_lr)*(1 + np.cos(cos_inner))



def make_env(slice_df, name):
    # Rename the column to match what the reward function expects
    if 'market_regime_code' not in slice_df.columns and 'market_regime_feature' in slice_df.columns:
        slice_df = slice_df.copy()
        slice_df['market_regime_code'] = slice_df['market_regime_feature']
    
    base = TradingEnv(
        slice_df,
        positions=[-1, 0, 1],  
        trading_fees=0.00005, 
        reward_function=risk_reward,
        windows=24,
        name=name
    )
    return AdaptiveRiskLimit(base, base_leverage=1.8, volatility_window=144)

# Custom learning rate schedule
def lr_schedule(progress):
    base_lr = 5e-5
    if progress < 0.3:
        return base_lr * (1 + 2*(0.3 - progress))  # High LR early
    else:
        return base_lr * (1 - (progress - 0.3)/0.7)  # Decay LR later
from stable_baselines3 import PPO


def ent_schedule(progress):
    # Higher starting entropy for much stronger exploration
    start_ent = 0.1  # Even higher (from 0.05)
    end_ent = 0.001   
    
    # Slower decay with quadratic falloff
    decay = progress**1.5  # Power > 1 makes it fall off slower initially
    
    return end_ent + (start_ent - end_ent) * decay  

def analyze_feature_importance(train_df, val_df, feature_cols):
    """Analyze and rank features by importance using correlation with returns"""
    print("Feature importance analysis:")
    
    # Calculate forward returns
    train_returns = np.log(train_df['close']).diff().shift(-1)
    
    # Calculate absolute correlation with future returns
    correlations = {}
    for feature in feature_cols:
        corr = abs(train_df[feature].corr(train_returns))
        if not np.isnan(corr):
            correlations[feature] = corr
    
    # Sort by correlation strength
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Print top and bottom features
    print("\nTop 15 most important features:")
    for feature, corr in sorted_features[:15]:
        print(f"  {feature}: {corr:.4f}")
    
    print("\nBottom 15 least important features:")
    for feature, corr in sorted_features[-15:]:
        print(f"  {feature}: {corr:.4f}")
    
    # Only keep features with correlation above threshold
    threshold = 0.01
    selected_features = [f for f, c in correlations.items() if c > threshold]
    
    print(f"\nKeeping {len(selected_features)}/{len(feature_cols)} features with correlation > {threshold}")
    
    return selected_features




def add_advanced_regime_features(df):
    """Add more sophisticated market regime features"""
    # Volatility regimes - multiple timeframes
    for window in [12, 36, 72]:
        vol = np.log(df['close']).diff().rolling(window).std()
        df[f'vol_regime_{window}h_feature'] = pd.qcut(
            vol, 5, labels=False, duplicates='drop').astype(float)
    
    # Trend strength using ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx_feature'] = adx['ADX_14']
    df['trend_strength_feature'] = df['adx_feature'] / 100.0
    
    # Momentum regime
    mom = df['close'].pct_change(24)
    df['momentum_regime_feature'] = pd.qcut(
        mom.rolling(72).mean(), 5, labels=False, duplicates='drop').astype(float)
    
    # Volume regime
    rel_vol = df['volume'] / df['volume'].rolling(72).mean()
    df['volume_regime_feature'] = pd.qcut(
        rel_vol, 5, labels=False, duplicates='drop').astype(float)
    
    # Combine regimes into a composite feature
    df['composite_regime_feature'] = (
        df['market_regime_feature'] * 0.4 + 
        df['momentum_regime_feature'] * 0.3 + 
        df['volume_regime_feature'] * 0.2 + 
        df['vol_regime_72h_feature'] * 0.1
    )
    
    return df

def clean_dataframe(df, cols):
    for col in cols:
        # Replace inf/-inf with NaN first
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaNs with column median for numeric features
        if df[col].dtype.kind in 'ifc':  # integer, float or complex
            median = df[col].median()
            # If median is NaN, use 0
            if pd.isna(median):
                median = 0
            df[col] = df[col].fillna(median)
    return df


df = add_advanced_regime_features(df)
results = []
equity_curve = []

for fold, (train_df, val_df) in enumerate(make_walk_forward_splits(df)):
    print(f"\n============= FOLD {fold} =============")
    
    # –– Escalado robusto SOLO con datos de entrenamiento
    feature_cols = [c for c in train_df.columns if c.endswith("_feature")]
    # Define target (retornos futuros)
    future_returns = np.log(train_df['close']).diff().shift(-1).fillna(0)

    # Selección recursiva de características 
    selector = RFE(RandomForestRegressor(n_estimators=100), n_features_to_select=10)
    selector.fit(train_df[feature_cols], future_returns)

    # Solo mantener características importantes
    selected_features = [f for f, selected in zip(feature_cols, selector.support_) if selected]
    # Keep only the base columns and selected features
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    required_cols = ['market_regime_code']
    for col in required_cols:
        if col in train_df.columns and col not in base_cols and col not in selected_features:
            selected_features.append(col)

    train_df = train_df[base_cols + selected_features]
    val_df = val_df[base_cols + selected_features]
    train_df = train_df[base_cols + selected_features]
    val_df = val_df[base_cols + selected_features]
    
    # Check for low variance ONLY on the selected features
    low_var = train_df[selected_features].var() < 1e-4
    drop_cols = low_var[low_var].index.tolist()

    if drop_cols:
        print(" Drop features:", drop_cols)
        train_df = train_df.drop(columns=drop_cols)
        val_df = val_df.drop(columns=drop_cols)
        # Update selected features to exclude dropped columns
        selected_features = [f for f in selected_features if f not in drop_cols]
    

    train_df = clean_dataframe(train_df, train_df.columns)
    val_df = clean_dataframe(val_df, val_df.columns)

    # Apply scaling to the final set of selected features
    scaler = RobustScaler(quantile_range=(25, 75))
    train_df[selected_features] = scaler.fit_transform(train_df[selected_features])
    val_df[selected_features] = scaler.transform(val_df[selected_features])

    # –– ENVs
    train_env = VecNormalize(
        DummyVecEnv([lambda: make_env(train_df, f"train{fold}")]),
        norm_obs=True, norm_reward=False)

    val_env   = VecNormalize(
        DummyVecEnv([lambda: make_env(val_df, f"val{fold}")]),
        norm_obs=True, norm_reward=False, training=False)
               # congelado
    # 2) copia las estadísticas de observación
    val_env.obs_rms   = deepcopy(train_env.obs_rms)
    val_env.clip_obs  = train_env.clip_obs         # mismo límite

    # 3) desactiva entrenamiento por si acaso
    val_env.training = False

    
    policy_kwargs = dict(
        lstm_hidden_size=32,       # Tamaño moderado
        n_lstm_layers=1,           # Una sola capa LSTM
        net_arch=dict(
            pi=[32],               # Red muy pequeña post-LSTM para actor
            vf=[64]                # Red pequeña para el crítico
        ),
        activation_fn=torch.nn.Tanh,  # Tanh a veces funciona mejor para secuencias
        ortho_init=True,
        enable_critic_lstm=True,   # Compartir LSTM entre actor y crítico
    )
    
    INIT_ENT = ent_schedule(1.0)      # valor al principio (progress=1)

    # –– Modelo nuevo por fold
    # 3. More conservative PPO parameters
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        policy_kwargs=policy_kwargs,
        env=train_env,
        n_steps=128,               # Menor que 512
        batch_size=32,
        n_epochs=3,                # Menor que 4
        learning_rate=1e-4,        # Menor que 2e-4
        ent_coef=0.2,              # Mayor que 0.1 (más exploración)
        gamma=0.985,               # Ligeramente menor descuento
        gae_lambda=0.92,
        max_grad_norm=0.3,         # Más restrictivo
        verbose=0,
        device="cpu",
        tensorboard_log="logs_recurrent"
    )


    print(model.clip_range,
      model.clip_range_vf,
      model.vf_coef,
      model.target_kl,
      model.ent_coef_schedule if hasattr(model, "ent_coef_schedule") else model.ent_coef)



    callbacks = CallbackList([
        EntropyDecay(ent_schedule, total_ts=800_000),  # Double training time
        ActionTrackingCallback(val_env, check_freq=10000),
        EarlyStopKL(kl_threshold=1e-4,  # Lower threshold
                    patience=10,        # More patience
                    warmup_rollouts=25, # More warmup
                    min_timesteps=600_000,  # Higher min training
                    verbose=1),
        DetailedLoggingCallback(verbose=1, log_freq=20000),
        ValidationPerformanceCallback(val_env, check_freq=25000),
        PeriodicValidation(val_env,
                        every_ts=50_000,
                        save_path=f"best_fold{fold}.zip",
                        metric="sharpe",
                        verbose=1)
    ])
    model.learn(total_timesteps=800_000,  # Double the training time
                callback=callbacks)

    train_hist = train_env.get_attr("historical_info")[0]
    fold_metrics = calculate_metrics(train_hist)
    print(f"Fold {fold} | sharpe: {fold_metrics['annualized_sharpe']:.2f}")
    print(f"Fold {fold} | win-rate: {fold_metrics['win_rate']:.2%}")
    print(f"Fold {fold} | profit_factor: {fold_metrics['profit_factor']:.2f}")
    print(f"Fold {fold} | max_DD_dur.: {fold_metrics['max_dd_duration']} pasos")
    print(f"Fold {fold} | bull perf.: {fold_metrics['regime_performance']['bull']:.2%}")
    print(f"Fold {fold} | bear perf.: {fold_metrics['regime_performance']['bear']:.2%}")

    

    

    # copia de estadísticas para que val_env use la misma normalización
    val_env.obs_rms   = train_env.obs_rms
    val_env.ret_rms   = train_env.ret_rms
    val_env.clip_obs  = train_env.clip_obs
    val_env.clip_reward = train_env.clip_reward

    # –– Validación determinista
    obs = val_env.reset()
    done = False
    valuations = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, dones, infos = val_env.step(action)
        done = dones[0] or infos[0].get("time_limit", False)  # << añade esto
        valuations.append(infos[0]["portfolio_valuation"])

    val_return = valuations[-1] / valuations[0] - 1
    max_dd = 1 - np.min(np.array(valuations) / np.maximum.accumulate(valuations))
    results.append({"fold": fold, "val_return": val_return, "max_dd": max_dd})
    equity_curve.extend(valuations)
    val_hist = val_env.get_attr("historical_info")[0]
    fold_metrics = calculate_metrics(val_hist)
    print(f"FOLD {fold} extra metrics:", fold_metrics)
# –– agrega métricas globales
import pandas as pd, numpy as np
res_df = pd.DataFrame(results)
if res_df.empty:
    print("❗ No se generó ningún split. ¿Tienes suficiente histórico?")
    sys.exit()

print("Retorno medio:", res_df["val_return"].mean())
print("Max DD medio:", res_df["max_dd"].mean())
print("\nResultados walk‑forward:")
print(res_df)
print("Retorno medio:", res_df["val_return"].mean())
print("Max DD medio:", res_df["max_dd"].mean())

# Al final del script, después de entrenar todos los folds:
# Implementa una estrategia de ensemble



def ensemble_predict(models, obs, weights=None):
    """Combina predicciones de varios modelos con ponderación opcional"""
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    actions = []
    for model, weight in zip(models, weights):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action * weight)
    
    return np.sum(actions, axis=0)

# Cargar los mejores modelos de cada fold
best_models = []
def calculate_ensemble_weights(results, min_weight=0.1):
    """Calculate more balanced ensemble weights"""
    # Extract metrics based on validation results
    metrics = []
    for result in results:
        # Create a composite score from multiple metrics
        sharpe = result.get('val_sharpe', 0)
        ret = result.get('val_return', 0)
        dd = result.get('max_dd', 1)  # Lower is better
        
        # Custom score formula that balances returns and risk
        score = (sharpe * 0.4) + (ret * 100 * 0.4) - (dd * 100 * 0.2)
        # Ensure minimum positive score for weight calculation
        score = max(0.1, score)
        metrics.append(score)
    
    # Convert to numpy array
    raw_weights = np.array(metrics)
    
    # Apply softmax-like normalization with temperature
    temp = 0.5  # Lower temperature = more balanced weights
    exp_weights = np.exp(raw_weights / temp)
    weights = exp_weights / exp_weights.sum()
    
    # Enforce minimum weight
    if min_weight > 0:
        low_idx = weights < min_weight
        if np.any(low_idx):
            # Redistribute excess weight from high to low
            n_low = low_idx.sum()
            shortfall = n_low * min_weight - weights[low_idx].sum()
            weights[~low_idx] -= shortfall / (~low_idx).sum()
            weights[low_idx] = min_weight
    
    return weights

# Replace your ensemble weight calculation with:
weights = calculate_ensemble_weights(results, min_weight=0.10)

exit()
# Paso 5: Generar y guardar señales para Freqtrade
obs, _ = env.reset()
signals = []
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    signals.append((info['date'], action))

signals_df = pd.DataFrame(signals, columns=["date", "action"])
signals_df.to_csv("user_data/signals/BTC_USDT.csv", index=False)
print("\nSeñales guardadas en user_data/signals/BTC_USDT.csv")
