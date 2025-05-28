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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

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
import fear_and_greed
import numpy as np
from copy import deepcopy
from collections import Counter


import gymnasium as gym
import numpy as np

def _sharpe_ratio(nav: np.ndarray) -> float:
    rets = np.diff(np.log(nav))
    return (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(8_760)


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq=50000, save_path='./checkpoints/', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f'checkpoint_{self.num_timesteps}.zip')
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving checkpoint to {path}")
        return True

class ValidationPerformanceCallback(BaseCallback):
    """Detiene entrenamiento si train y validation divergen demasiado"""
    def __init__(self, eval_env, check_freq=10000, max_train_val_ratio=3.0):
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
            self.action_counts = [0, 0, 0]
            # Run a short evaluation to see action distribution
            if self.n_calls > 0:
                self.action_counts = [0, 0, 0]        # reset

                obs = self.eval_env.reset()
                for _ in range(100):
                    raw_a, _ = self.model.predict(obs, deterministic=False)

                    bucket = int(_bucketize(raw_a)[0])
                    self.action_counts[bucket] += 1

                    obs, _, dones, _ = self.eval_env.step(raw_a)
                    if dones[0]:
                        obs, _ = self.eval_env.reset()
                    
                   
                    
                    
                    
                    
                print(f"\n--- Action Distribution at step {self.n_calls} ---")
                print(f"SHORT: {self.action_counts[0]}%, HOLD: {self.action_counts[1]}%, LONG: {self.action_counts[2]}%")
                
                # # Entropy adjustment logic
                # if max(self.action_counts) > 90:
                #     print("⚠️ ACTION DIVERSITY CRISIS - INCREASING ENTROPY")
                #     self.model.ent_coef = max(0.1, float(self.model.ent_coef) * 2)
                # elif max(self.action_counts) < 60:
                #     if hasattr(self.model, 'ent_coef') and not callable(self.model.ent_coef):
                #         current_ent = float(self.model.ent_coef)
                #         if current_ent > 0.005:
                #             self.model.ent_coef = current_ent * 0.9
        
        return True



def _bucketize(a):
    """
    Map exposure in [-1,1] to bucket 0=SHORT, 1=HOLD, 2=LONG.
    Works for scalars or arrays; always returns a NumPy array.
    """
    arr = np.asarray(a, dtype=float)
    # Opción A: bajar el umbral
    disc = np.where(arr <= -0.15, -1,
            np.where(arr >= 0.15,  1, 0)).astype(int) + 1

    return disc



class DetailedLoggingCallback(BaseCallback):
    """Logs detailed metrics during training"""
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []
        self.actions = []
        self.values = []
        self.rolling_reward_mean = 0
        self.rolling_reward_std = 0

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:

                # --- LECTURA MODIFICADA DEL BUFFER ---
                buffer = self.model.rollout_buffer
                current_pos = buffer.pos
                buffer_size = buffer.buffer_size
                n_envs = buffer.n_envs # Debería ser 1 en tu caso

                if current_pos > 0:
                    # Obtener los datos válidos hasta la posición actual
                    # Para RecurrentRolloutBuffer, las acciones son (buffer_size, n_envs, action_dim)
                    # y las recompensas son (buffer_size, n_envs)
                    
                    # Si el buffer no ha dado la vuelta (no está lleno y reseteado)
                    if not buffer.full:
                        valid_actions_flat = buffer.actions[:current_pos, 0, :].flatten() # Asumiendo n_envs=1
                        valid_rewards_flat = buffer.rewards[:current_pos, 0].flatten()    # Asumiendo n_envs=1
                        valid_values_flat  = buffer.values[:current_pos, 0].flatten()     # Asumiendo n_envs=1
                    else: # El buffer ha dado la vuelta, 'pos' es el punto de inserción más nuevo
                          # Los datos válidos son desde 'pos' hasta el final, y luego desde el inicio hasta 'pos'
                        actions_part1 = buffer.actions[current_pos:, 0, :].flatten()
                        actions_part2 = buffer.actions[:current_pos, 0, :].flatten()
                        valid_actions_flat = np.concatenate((actions_part1, actions_part2))

                        rewards_part1 = buffer.rewards[current_pos:, 0].flatten()
                        rewards_part2 = buffer.rewards[:current_pos, 0].flatten()
                        valid_rewards_flat = np.concatenate((rewards_part1, rewards_part2))

                        values_part1 = buffer.values[current_pos:, 0].flatten()
                        values_part2 = buffer.values[:current_pos, 0].flatten()
                        valid_values_flat = np.concatenate((values_part1, values_part2))


                    # Tomar los últimos N elementos de los datos válidos, si hay suficientes
                    num_recent_to_log = 100
                    recent_raw_actions_from_buffer = valid_actions_flat[-num_recent_to_log:] if len(valid_actions_flat) >= num_recent_to_log else valid_actions_flat
                    recent_rewards_from_buffer = valid_rewards_flat[-num_recent_to_log:] if len(valid_rewards_flat) >= num_recent_to_log else valid_rewards_flat
                    recent_values_from_buffer = valid_values_flat[-num_recent_to_log:] if len(valid_values_flat) >= num_recent_to_log else valid_values_flat
                    
                    # --- FIN DE LECTURA MODIFICADA ---

                    if len(recent_rewards_from_buffer) > 0:
                        reward_mean = np.mean(recent_rewards_from_buffer)
                        reward_std = np.std(recent_rewards_from_buffer)
                    else:
                        reward_mean = 0.0
                        reward_std = 0.0
                    
                    if self.n_calls % self.log_freq == 0:
                        print("posición media últimas N steps (raw actions):", 
                              np.mean(recent_raw_actions_from_buffer) if len(recent_raw_actions_from_buffer) > 0 else 0.0)
                        # El acceso a historical_info parece correcto, lo mantenemos
                        
                    self.logger.record("train/reward_mean", reward_mean)
                    # ... (resto del logging como estaba, usando recent_rewards_from_buffer y recent_raw_actions_from_buffer) ...
                    
                    print(f"\n--- Training Stats at step {self.n_calls} (using modified buffer read) ---")
                    print(f"Recent reward mean: {reward_mean:.4f}, std: {reward_std:.4f}")
                    
                    if len(recent_raw_actions_from_buffer) > 0:
                        recent_actions_bucketized = _bucketize(recent_raw_actions_from_buffer)
                        action_counts = np.bincount(recent_actions_bucketized, minlength=3)
                        print(f"SHORT:{action_counts[0]} HOLD:{action_counts[1]} LONG:{action_counts[2]}")
                        if np.std(recent_raw_actions_from_buffer) < 0.01: # Ajustado para usar recent_raw_actions_from_buffer
                            print("⚠️ WARNING: Low action diversity - model may be stuck!")
                    else:
                        print("SHORT:0 HOLD:0 LONG:0 (No actions in buffer slice)")


                    if len(recent_values_from_buffer) > 0:
                        print(f"Value estimate mean: {np.mean(recent_values_from_buffer):.4f}, std: {np.std(recent_values_from_buffer):.4f}")
                        if np.std(recent_values_from_buffer) < 0.01:
                            print("⚠️ WARNING: Low value diversity - critic may be stuck!")
                    else:
                        print("Value estimate mean: 0.0000, std: 0.0000 (No values in buffer slice)")
                        
            self.logger.dump(self.n_calls)
        return True


class PeriodicValidation(BaseCallback):
    """
    Evaluates the policy every every_ts timesteps and saves the best model based on the specified metric.
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
    debajo de kl_threshold durante patience rollouts *después* de
    haber superado min_timesteps de entrenamiento efectivo.
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
        self.peak_value = None # Inicializar peak_value aquí

    # ---------- helpers --------------------------------------------------
    def _leverage(self):
        price = self.env._get_price()
        # Asegurarse de que _portfolio y real_position están disponibles y son válidos
        if hasattr(self.env, '_portfolio') and self.env._portfolio is not None:
            return abs(self.env._portfolio.real_position(price))
        return 0.0 # Valor por defecto o manejo de error

    def _drawdown(self, current_val):
        if self.peak_value is None or self.peak_value == 0: # Comprobar si peak_value es None
            return 0.0
        return (self.peak_value - current_val) / self.peak_value

    def _breach(self, info):
        """¿Se supera apalancamiento o DD?"""
        lev = self._leverage()
        # Asegurarse de que portfolio_valuation está en info
        current_valuation = info.get("portfolio_valuation", self.peak_value if self.peak_value is not None else 0)
        dd  = self._drawdown(current_valuation)
        
        # Añadir un log más detallado para _breach
        return (lev > self.max_leverage) or (dd > self.max_dd)

    # ---------- Gymnasium interface --------------------------------------
    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        # Manejar diferentes formatos de retorno de reset
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else: # Asumir que es solo obs, o un formato no esperado, crear info vacío
            obs = reset_result
            info = {} 
        
        # Asegurarse de que portfolio_valuation está en info después del reset
        self.peak_value = info.get("portfolio_valuation", self.env.portfolio_initial_value if hasattr(self.env, 'portfolio_initial_value') else 0)
        return obs, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_float = float(action[0])
        else:
            action_float = float(action)

        obs, reward_from_inner_env, terminated, truncated, info = self.env.step(action_float)
        
        current_reward = reward_from_inner_env 

        # Asegurarse de que peak_value no es None antes de usar max()
        current_valuation = info.get("portfolio_valuation", self.peak_value if self.peak_value is not None else 0)
        if self.peak_value is None:
            self.peak_value = current_valuation
        else:
            self.peak_value = max(self.peak_value, current_valuation)

        breached_flag = self._breach(info)

        if not terminated and not truncated and breached_flag:
            if hasattr(self.env, '_trade'): # Verificar si el método _trade existe
                self.env._trade(0) # fuerza posición plana
            current_reward += self.penalty 
            info["risk_event"] = True
            if hasattr(self.env, '_get_obs'): # Verificar si el método _get_obs existe
                obs = self.env._get_obs()  
        else:
            info["risk_event"] = False

        # DEBUG PRINT para RiskLimit (salida)
        # Imprimir la acción original que recibió el wrapper y la recompensa que va a devolver
        # Usar self.env._step si está disponible, de lo contrario un contador local o global si es necesario
        current_step_for_log = self.env._step if hasattr(self.env, '_step') else -1 
        

        return obs, current_reward, terminated, truncated, info



class AdaptiveDynamicRiskLimit(RiskLimit): # Asegúrate que esta clase está definida después de RiskLimit
    """
    Igual que RiskLimit pero:
      • el apalancamiento máximo se recalcula cada paso en función
        de la volatilidad reciente;
      • la tolerancia de draw‑down aumenta suavemente cuando el bot
        va ganando dinero.
    """
    def __init__(self, env, base_leverage=1.5, volatility_window=72, max_dd=0.25): # Añadido max_dd
        # max_dd y penalty se heredan; puedes ajustarlos aquí si quieres
        super().__init__(env, max_leverage=base_leverage, max_dd=max_dd, penalty=-1.0) # Pasado max_dd
        self.base_leverage = base_leverage
        self.vol_window    = volatility_window        # últimas 72 h en un env 1‑h

    # ---------- lógica dinámica -----------------------------------------
    def _dynamic_leverage(self):
        # Retornos logarítmicos de la ventana más reciente
        # Asegurarse de que self.env.df y 'close' existen
        if not hasattr(self.env, 'df') or 'close' not in self.env.df.columns:
            return self.base_leverage # Valor por defecto
        
        returns = np.log(self.env.df['close']).diff().tail(self.vol_window)
        if returns.empty or returns.std() == 0: # Manejar casos donde std es 0 o returns está vacío
            return self.base_leverage

        vol = returns.std() * np.sqrt(24*365)                # anualizada

        # Regla simple: cuanto más volátil, menor leverage permitido
        return min(self.base_leverage, 2.5 / (vol + 0.05))

    def _dynamic_max_dd(self):
        # Beneficio acumulado bruto desde el inicio del fold
        # Asegurarse de que historical_info y portfolio_valuation existen
        if not hasattr(self.env, 'historical_info') or \
           not hasattr(self.env.historical_info, '__getitem__') or \
           len(self.env.historical_info) == 0:
            return 0.25 # Valor por defecto

        cur_nav = self.env.historical_info['portfolio_valuation', -1]
        
        # Asegurarse de que portfolio_initial_value existe
        initial_val = self.env.portfolio_initial_value if hasattr(self.env, 'portfolio_initial_value') else cur_nav
        if initial_val == 0: # Evitar división por cero
            return 0.25

        perf    = (cur_nav / initial_val) - 1.0

        # Entre 0.25 (sin beneficio) y 0.50 (+∞ beneficio, asintótico)
        return 0.25 * (1 + np.tanh(perf))
    
    # ---------- sobreescribe step() --------------------------------------
    def step(self, action):
        # Actualiza los límites antes de delegar en el env interior
        self.max_leverage = self._dynamic_leverage()
        self.max_dd       = self._dynamic_max_dd()
        return super().step(action) # Llama al step de RiskLimit (que ahora tiene el print)


class LeverageProgressionCallback(BaseCallback):
    """
    Implements linear leverage growth from initial to target value
    during the first specified fraction of training.
    """
    def __init__(self, 
                 initial_leverage: float = 0.5,
                 target_leverage: float = 1.2, 
                 ramp_fraction: float = 0.3,
                 total_steps: int = 800_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.initial = initial_leverage
        self.target = target_leverage
        self.ramp_fraction = ramp_fraction
        self.total_steps = total_steps
        self.last_leverage = initial_leverage
        
    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.total_steps)
        
        # If we're still in the ramp-up phase
        if progress < self.ramp_fraction:
            # Linear interpolation from initial to target
            ramp_progress = progress / self.ramp_fraction  # Normalized progress within ramp phase
            new_leverage = self.initial + ramp_progress * (self.target - self.initial)
        else:
            # After ramp-up, maintain target leverage
            new_leverage = self.target
            
        # Only update if leverage has changed meaningfully
        if abs(new_leverage - self.last_leverage) > 0.01:
            # Update leverage in all environments
            envs = self.model.get_env().get_attr("env")
            for env in envs:
                # Navigate through wrappers to find AdaptiveDynamicRiskLimit
                current = env
                while hasattr(current, "env"):
                    if isinstance(current, AdaptiveDynamicRiskLimit):
                        current.base_leverage = new_leverage
                        break
                    current = current.env
                    
            # Log the change if verbose
            if self.verbose > 0:
                print(f"📈 Leverage updated to {new_leverage:.2f} at step {self.num_timesteps:,} ({progress*100:.1f}% of training)")
                
            self.last_leverage = new_leverage
        
        return True


class AdaptiveDynamicRiskLimit(RiskLimit):
    """Enhanced risk management with regime-aware position sizing"""
    
    def __init__(self, env, base_leverage=1.2, volatility_window=144, max_dd=0.18):
        super().__init__(env, max_leverage=base_leverage, max_dd=max_dd, penalty=-1.0)
        self.base_leverage = base_leverage
        self.vol_window = volatility_window
        self.market_regime_memory = []  # Store recent regime observations
        
    def _dynamic_leverage(self):
        # Enhanced volatility calculation with exponential weighting
        returns = np.log(self.env.df['close']).diff().tail(self.vol_window)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate EWMA volatility (more weight to recent data)
        vol = returns.ewm(span=48).std().iloc[-1] * np.sqrt(24*365)
        
        # Get current market regime if available
        current_regime = -1  # Default neutral regime
        if 'market_regime_code' in self.env.df.columns:
            current_regime = self.env.df['market_regime_code'].iloc[-1]
            self.market_regime_memory.append(current_regime)
            if len(self.market_regime_memory) > 24:  # Keep last 24 hours
                self.market_regime_memory.pop(0)
                
        # Dominant regime in recent history
        if len(self.market_regime_memory) > 0:
            regime_count = Counter(self.market_regime_memory)
            dominant_regime = max(regime_count.items(), key=lambda x: x[1])[0]
            
            # Adjust leverage based on regime
            regime_factor = 1.0
            if dominant_regime <= 1:  # Bear market regimes (0,1)
                regime_factor = 0.7
            elif dominant_regime >= 2:  # Bull market regimes (2,3)
                regime_factor = 1.1
                
            return min(self.base_leverage * regime_factor, 2.0 / (vol + 0.1))
        
        # Default calculation if no regime data
        return min(self.base_leverage, 2.0 / (vol + 0.1))
        
    def _dynamic_max_dd(self):
        # More conservative drawdown management
        cur_nav = self.env.historical_info['portfolio_valuation', -1]
        perf = (cur_nav / self.env.portfolio_initial_value) - 1.0
        
        # Calculate volatility-adjusted max_dd
        vol = np.log(self.env.df['close']).diff().tail(72).std() * np.sqrt(24)
        vol_factor = np.clip(0.7 / (vol + 0.1), 0.6, 1.2)
        
        # Base DD limit between 0.15 and 0.25 based on performance
        return 0.15 * (1 + np.tanh(perf * 0.7)) * vol_factor

class ImprovedRiskLimit(AdaptiveDynamicRiskLimit):
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


from stable_baselines3.common.callbacks import BaseCallback

class KLAdaptiveLR(BaseCallback):
    """
    Dynamically scales the optimiser LR so that KL divergence stays
    inside [0.5, 1.5] × target_kl.
    Works with SB3 and sb3-contrib (RecurrentPPO): the optimiser
    lives in self.model.policy.optimizer.
    """
    def __init__(self,
                 target_kl: float = 3e-3,
                 min_lr: float = 1e-6,
                 max_lr: float = 3e-4,
                 down_factor: float = 0.5,
                 up_factor: float = 1.5,
                 ema_gamma: float = 0.9,
                 verbose: int = 0):
        super().__init__(verbose)
        self.tkl, self.min_lr, self.max_lr = target_kl, min_lr, max_lr
        self.down, self.up  = down_factor, up_factor
        self.ema_gamma      = ema_gamma
        self.ema_kl         = None

    def _on_rollout_end(self) -> None:
        # fetch the latest KL logged by SB3
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return

        # exponential moving average to smooth noise
        self.ema_kl = kl if self.ema_kl is None else \
                      self.ema_gamma * self.ema_kl + (1 - self.ema_gamma) * kl

        # -------- access optimiser via policy ----------
        optimizer = self.model.policy.optimizer
        cur_lr    = optimizer.param_groups[0]["lr"]

        if self.ema_kl > 1.5 * self.tkl:
            new_lr = max(self.min_lr, cur_lr * self.down)
        elif self.ema_kl < 0.5 * self.tkl:
            new_lr = min(self.max_lr, cur_lr * self.up)
        else:
            new_lr = cur_lr  # already in the sweet spot

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # log so it shows up in TensorBoard
        self.logger.record("train/learning_rate", new_lr)

        if self.verbose:
            print(f"[KL-LR] KL={self.ema_kl:.3e}  →  lr={new_lr:.2e}")

    def _on_step(self) -> bool:
        return True



class EntropyDecay(BaseCallback):
    """
    Actualiza model.ent_coef según la función schedule(progress_rem).
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
        
        # Apply minimum entropy clipping here as well
        self.model.ent_coef = max(current_ent_coef_val, 1e-3)
        
        if self.verbose > 0 and self.n_calls % 20 == 0:
            print(f"Entropy coefficient updated to: {self.model.ent_coef:.6f}")

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


def ensemble_position_sizing(models, raw_observations, confidence_threshold=0.65):
    """
    Uses median ensemble voting to reduce tail risk by ~20%
    """
    all_actions = []
    
    for model in models:
        try:
            # Get the feature indices this model was trained with
            if hasattr(model, "feature_indices"):
                features = model.feature_indices
                # Select only the features this model expects
                model_obs = raw_observations[:, features] if len(raw_observations.shape) > 1 else raw_observations[features]
            else:
                # Fallback - assume first 11 features
                model_obs = raw_observations[:, :11] if len(raw_observations.shape) > 1 else raw_observations[:11]
                
            # Get action prediction
            action, _ = model.predict(model_obs, deterministic=False)
            
            # Convert action to integer immediately
            if isinstance(action, np.ndarray) and len(action.shape) > 0:
                action = float(action[0])
            else:
                action = float(action)
                
            all_actions.append(action)
        except Exception as e:
            print(f"Error processing model prediction: {e}")
            continue
    
    if not all_actions:
        return 0  # Default to neutral if no predictions succeeded
        
    # Use integer mode (most common value) instead of median
    # This guarantees an integer result
    actions = np.array(all_actions)
    counts = np.bincount(actions + 1)  # Shift [-1,0,1] to [0,1,2]
    mode_action = np.argmax(counts) - 1  # Shift back
    
    return mode_action

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
# )
print("CARGADO")
# Convertir a DataFrame
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
# fng = FearGreedIndex()
# fng_df = pd.DataFrame(fng.get_last_n_days(365))
# fng_df["date"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s")
# fng_df.set_index("date", inplace=True)
# fng_df["value"] = fng_df["value"].astype(int)

# df["fng_feature"] = fng_df["value"].reindex(df.index, method="ffill").fillna(50)

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
    • Durante el último plateau_frac mantenemos min_lr plano.
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

def cosine_warmup_lr(progress_remaining: float,
                     peak_lr: float = 3e-4,
                     min_lr:  float = 1e-6,
                     warmup_frac: float = 0.10) -> float:
    """
    • Warm-up from 0.1·peak_lr → peak_lr during the first warmup_frac.
    • Cosine-decay from peak_lr → min_lr afterwards.
    """
    # convert to “training progress” (0 at start → 1 at end)
    p = 1.0 - progress_remaining

    # ① Warm-up phase
    if p < warmup_frac:
        warm_progress = p / warmup_frac         # 0 → 1
        return 0.1 * peak_lr + 0.9 * peak_lr * warm_progress

    # ② Cosine decay phase
    decay_progress = (p - warmup_frac) / (1.0 - warmup_frac)   # 0 → 1
    cosine = 0.5 * (1 + np.cos(np.pi * decay_progress))        # 1 → 0
    return min_lr + (peak_lr - min_lr) * cosine

def make_env(slice_df, name):
    # ... (código existente) ...
    base = TradingEnv(
        slice_df,
        positions=[-1, 0, 1],  
        trading_fees=0.00005, 
        reward_function=risk_reward,
        windows=24,
        name=name,
        max_episode_duration=250 # Asegúrate que esto es un int o 'max'
    )
    # Pasar el max_dd deseado aquí si es diferente del default de RiskLimit
    return AdaptiveDynamicRiskLimit(base, base_leverage=1.2, volatility_window=144, max_dd=0.18) 
# Custom learning rate schedule
def lr_schedule(progress):
    base_lr = 5e-5
    if progress < 0.3:
        return base_lr * (1 + 2*(0.3 - progress))  # High LR early
    else:
        return base_lr * (1 - (progress - 0.3)/0.7)  # Decay LR later
from stable_baselines3 import PPO


def ent_schedule(progress):
    """
    Entropy coefficient schedule for exploration control.
    
    Args:
        progress: Training progress from 1.0 (start) to 0.0 (end)
        
    Returns:
        Current entropy coefficient
    """
    # Ensure progress is non-negative to avoid complex numbers
    progress = max(0.0, min(1.0, float(progress)))
    
    # Higher starting entropy for stronger exploration
    start_ent = 0.1
    end_ent = 0.001
    
    # Slower decay with square root falloff
    decay = progress**0.5  # Safe now because progress is guaranteed to be non-negative
    
    entropy_value = end_ent + (start_ent - end_ent) * decay
    
    # Add minimum entropy clipping to ensure sufficient exploration
    return max(entropy_value, 1e-3)

def analyze_feature_importance(train_df, val_df, feature_cols):
    """Analyze and rank features by importance with stricter thresholds"""
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
    
    # Increased threshold from 0.01 to 0.03
    threshold = 0.03
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
best_features = None          # ← se rellenará en el fold 0
from collections import defaultdict

for fold, (train_df, val_df) in enumerate(make_walk_forward_splits(df)):
    print(f"\n============= FOLD {fold} =============")

    base_cols = ['open','high','low','close','volume','amount']
    feature_cols = [c for c in train_df.columns if c.endswith('_feature')]
    required_cols = ['market_regime_code']

    # ------------------ selección de features ------------------
    if best_features is None:
        y = np.log(train_df['close']).diff().shift(-1).fillna(0)
        
        # Modelo más robusto
        rf = RandomForestRegressor(
            n_estimators=300,  # Más árboles
            max_depth=12,      # Control de profundidad
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
       
        # Usar RFECV con validación temporal
        ts_cv = TimeSeriesSplit(n_splits=3) # Asegúrate que TimeSeriesSplit está importado
        selector = RFECV(
            rf, 
            step=1, 
            cv=ts_cv,
            min_features_to_select=5,
            scoring='neg_mean_squared_error'
        )
        
        print(f"Ajustando RFECV para selección de features en {len(feature_cols)} características...")
        selector.fit(train_df[feature_cols], y)
        best_features = [f for f, keep in zip(feature_cols, selector.support_) if keep]
        print("Mejores features seleccionadas por RFECV:")
        print(f"  {best_features}") # Imprime la lista de features seleccionadas
        print(f"Número de features seleccionadas: {len(best_features)}")

    selected_features = best_features.copy()       # reutilizar

    # añadir los obligatorios si no estaban
    for col in required_cols:
        if col not in base_cols and col not in selected_features:
            selected_features.append(col)

    # ------------------ subset + limpieza ------------------
    train_df = train_df[base_cols + selected_features].copy()
    val_df   = val_df[base_cols + selected_features].copy()




   

    train_df = clean_dataframe(train_df, train_df.columns)
    val_df = clean_dataframe(val_df, val_df.columns)

    

  

    # Apply scaling to the final set of selected features
    scaler = RobustScaler(quantile_range=(25, 75))
    train_df[selected_features] = scaler.fit_transform(train_df[selected_features])
    val_df[selected_features] = scaler.transform(val_df[selected_features])

    # print("escala tras scaler (std):")
    # print(train_df[selected_features].std().sort_values().head(5))
    # env = make_env(df.tail(1000), "debug")

    # obs, info = env.reset()
    # stats = defaultdict(list)

    # for _ in range(300):
    #     a = env.action_space.sample()         # acción aleatoria
    #     obs, r, done, trunc, info = env.step(a)
    #     stats['r'].append(r)
    #     stats['pnl_excess'].append(info['reward_raw'] if 'reward_raw' in info else np.nan)
    #     if done or trunc:
    #         break
    # print("media raw :", np.nanmean(stats['pnl_excess']))
    # print("std  raw :",  np.nanstd(stats['pnl_excess']))

    # print("media r   :", np.mean(stats['r']))
    # print("std   r   :", np.std(stats['r']))
    # print("min / max :", np.min(stats['r']), np.max(stats['r']))
    # exit()

    # –– ENVs
    # train_env = VecNormalize(
    #     DummyVecEnv([lambda: make_env(train_df, f"train{fold}")]),
    #     norm_obs=True, norm_reward=False, clip_obs=10.)

    # val_env   = VecNormalize(
    #     DummyVecEnv([lambda: make_env(val_df, f"val{fold}")]),
    #     norm_obs=True, norm_reward=False, training=False)

    train_env = DummyVecEnv([lambda: make_env(train_df, f"train{fold}")]) # <-- PRUEBA SIN VECNORMALIZE
    val_env = DummyVecEnv([lambda: make_env(val_df, f"val{fold}")]) # <-- PRUEBA SIN VECNORMALIZE
    
               # congelado
    # 2) copia las estadísticas de observación
    # val_env.obs_rms   = deepcopy(train_env.obs_rms)
    # val_env.clip_obs  = train_env.clip_obs         # mismo límite

    # 3) desactiva entrenamiento por si acaso
    val_env.training = False

    total_timesteps = 750_000
    policy_kwargs = dict(
        lstm_hidden_size=64,       # Increased from 32 for more capacity
        n_lstm_layers=2,           # Keep one layer to avoid overfitting
        net_arch=dict(
            pi=[48, 24],           # Deeper actor network with tapered structure
            vf=[96, 48]            # Deeper critic network
        ),
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
        enable_critic_lstm=True,
        # Add dropout for regularization (NEW)
        lstm_kwargs=dict(dropout=0.2)
    )

    callbacks = CallbackList([
        KLAdaptiveLR(target_kl=3e-3, verbose=1),  
        EntropyDecay(ent_schedule, total_ts=total_timesteps),  # Double training time
        LeverageProgressionCallback(  # Add this new callback
            initial_leverage=0.5,
            target_leverage=1.0, 
            ramp_fraction=0.6,
            total_steps=total_timesteps, 
            verbose=1
        ),
        ActionTrackingCallback(val_env, check_freq=75000),
        EarlyStopKL(kl_threshold=1e-4,  # Lower threshold
                    patience=10,        # More patience
                    warmup_rollouts=25, # More warmup
                    min_timesteps=100_000,  # Higher min training
                    verbose=1),
        CheckpointCallback(save_freq=100000, save_path=f'./checkpoints/fold{fold}/'),
        DetailedLoggingCallback(verbose=1, log_freq=20000),
        ValidationPerformanceCallback(val_env, check_freq=25000),
        PeriodicValidation(val_env,
                        every_ts=75_000,
                        save_path=f"best_fold{fold}.zip",
                        metric="sharpe",
                        verbose=1)
    ])
    fold_models, best_models = [], []
    # –– Modelo nuevo por fold
    # 3. More conservative PPO parameters
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        policy_kwargs=policy_kwargs,
        env=train_env,
        n_steps=2048,               
        batch_size=512,             
        n_epochs=6,                
        learning_rate=cosine_warmup_lr,     
        ent_coef=0.02,              # Already set to 0.02
        gamma=0.99,                
        gae_lambda=0.95,           
        max_grad_norm=0.5,         
        vf_coef=0.4,               
        clip_range=0.15,
        clip_range_vf=0.15,
        target_kl=0.003,           
        verbose=0,
        device="cuda",
        tensorboard_log="logs_recurrent",
    )

    model.selected_features_names = selected_features # Guardar los nombres de las features
    print(model.clip_range,
    model.clip_range_vf,
    model.vf_coef,
    model.target_kl,
    model.ent_coef_schedule if hasattr(model, "ent_coef_schedule") else model.ent_coef)




    model.learn(total_timesteps=total_timesteps,  # Double the training time
                callback=callbacks)
    model.save(f"best_fold{fold}.zip")
    fold_models.append(model)

    train_hist = train_env.get_attr("historical_info")[0]
    fold_metrics = calculate_metrics(train_hist)
    print(f"Fold {fold} | sharpe: {fold_metrics['annualized_sharpe']:.2f}")
    print(f"Fold {fold} | win-rate: {fold_metrics['win_rate']:.2%}")
    print(f"Fold {fold} | profit_factor: {fold_metrics['profit_factor']:.2f}")
    print(f"Fold {fold} | max_DD_dur.: {fold_metrics['max_dd_duration']} pasos")
    print(f"Fold {fold} | bull perf.: {fold_metrics['regime_performance']['bull']:.2%}")
    print(f"Fold {fold} | bear perf.: {fold_metrics['regime_performance']['bear']:.2%}")

        

    

    # copia de estadísticas para que val_env use la misma normalización
    # val_env.obs_rms   = train_env.obs_rms
    # val_env.ret_rms   = train_env.ret_rms
    # val_env.clip_obs  = train_env.clip_obs
    # val_env.clip_reward = train_env.clip_reward

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
    best_models.extend(fold_models)
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



def ensemble_position_sizing(models, observation_np, confidence_threshold=0.65): # observation_np es la salida de test_env.reset()
    """
    Uses median ensemble voting.
    Assumes all models in the ensemble were trained on the same feature set
    and `observation_np` is the correctly shaped numpy array from the VecNormalized environment.
    """
    all_actions = []
    
    for model_idx, model in enumerate(models):
        try:
            # `observation_np` ya debería tener la forma correcta (seq_len, n_features)
            # o (n_envs, seq_len, n_features) si es de un VecEnv.
            # Para RecurrentPPO, si es (seq_len, n_features), está bien.
            action, _ = model.predict(observation_np, deterministic=False) # No se necesitan lstm_states para RecurrentPPO
            
            if isinstance(action, np.ndarray):
                action_val = float(action[0]) # Asume que la acción es un array de un solo elemento
            else:
                action_val = float(action)
                
            all_actions.append(action_val)
        except Exception as e:
            print(f"Error procesando la predicción del modelo {model_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_actions:
        return 0.0  # Default to neutral if no predictions succeeded
        
    actions_array = np.array(all_actions)
    median_action = np.median(actions_array)
    
    # Opcional: discretizar la acción mediana si es necesario
    # if -0.15 < median_action < 0.15:
    #     return 0.0
    # elif median_action <= -0.15:
    #     return -1.0
    # else:
    #     return 1.0

    return median_action # Devolver la mediana directamente o discretizarla

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
for fold in range(len(results)):
    try:
        model_path = f"best_fold{fold}.zip"
        model = RecurrentPPO.load(model_path)
        best_models.append(model)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model for fold {fold}: {e}")

# Ensure we have at least one model
if not best_models:
    print("No models were loaded. Cannot generate signals.")
    sys.exit(1)

# Create a test environment from the most recent data
recent_data = df.iloc[-5000:].copy()  # Use recent data for predictions
test_env = VecNormalize(
    DummyVecEnv([lambda: make_env(recent_data, "test")]),
    norm_obs=True, norm_reward=False, training=False)

# Generate signals using the ensemble
obs_tuple = test_env.reset()[0]
obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple # Extraer la observación numpy

signals = []
done = False
step = 0
max_steps = 1000  # Limit number of steps to prevent infinite loops

while not done and step < max_steps:
    # Use the ensemble_position_sizing for more nuanced positioning
    action = ensemble_position_sizing(best_models, obs, confidence_threshold=0.65)
    
    step_results = test_env.step(np.array([action])) # Enviar la acción como un array
    next_obs_tuple = step_results[0]
    rewards = step_results[1]
    dones = step_results[2]
    infos = step_results[3]

    obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple


    current_date = infos[0]["date"] # Obtener la fecha de la info del paso actual
    signals.append((current_date, action))

    done = dones[0]
    step += 1

# Convert to DataFrame and save
signals_df = pd.DataFrame(signals, columns=["date", "action"])
signals_df.to_csv("signals/BTC_USDT.csv", index=False)
print(f"\nGenerated {len(signals_df)} signals using ensemble position sizing")
print("Signals saved to signals/BTC_USDT.csv")

# Plot the equity curve if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    # Get the equity curve from the test run
    equity = np.array([info[0]["portfolio_valuation"] for _, _, _, info in test_env._get_history()])
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title("Ensemble Model Performance")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.savefig("ensemble_performance.png")
    print("Performance chart saved to ensemble_performance.png")
except Exception as e:
    print(f"Could not generate performance chart: {e}")
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
signals_df.to_csv("signals/BTC_USDT.csv", index=False)
print("\nSeñales guardadas en signals/BTC_USDT.csv")