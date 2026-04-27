import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


def _safe_log_return(current_value, previous_value) -> float:
    """Numerically stable one-step log return."""
    previous_value = float(previous_value)
    current_value = float(current_value)
    if previous_value <= 1e-10 or current_value <= 0:
        return 0.0
    return float(np.log(max(current_value / previous_value, 1e-10)))


def _market_log_return(history: 'History') -> float:
    """One-step market log return from the underlying close series."""
    close_now = history['data_close', -1]
    close_prev = history['data_close', -2]
    return _safe_log_return(close_now, close_prev)


import datetime
import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from collections import Counter

import tempfile, os
# BUG #11 FIX: Removed global warnings.filterwarnings("error") which was causing action-at-a-distance failures.


class Portfolio:
    """Track asset/fiat positions with margin/borrow interest support."""

    def __init__(self, asset: float, fiat: float,
                 interest_asset: float = 0, interest_fiat: float = 0) -> None:
        self.asset = asset
        self.fiat = fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat
    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat,
            - self.interest_asset * price,
            - self.interest_fiat
        ])
    def real_position(self, price):
        return (self.asset - self.interest_asset)* price / self.valorisation(price)
    def position(self, price):
        return self.asset * price / self.valorisation(price)
    def trade_to_position(self, position, price, trading_fees):
        # Repay interest
        current_position = self.position(price)
        interest_reduction_ratio = 1
        if (position <= 0 and current_position < 0):
            interest_reduction_ratio = min(1, position/current_position)
        elif (position >= 1 and current_position > 1):
            interest_reduction_ratio = min(1, (position-1)/(current_position-1))
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1-interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1-interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat
        
        # Proceed to trade
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade 
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)
    def update_interest(self, borrow_interest_rate):
        # BUG #5 FIX: Accumulate interest instead of replacing it.
        # Otherwise, the agent only pays for the last bar of leverage, not the duration.
        self.interest_asset += max(0, - self.asset) * borrow_interest_rate
        self.interest_fiat += max(0, - self.fiat) * borrow_interest_rate
    def __str__(self): return f"{self.__class__.__name__}({self.__dict__})"
    def describe(self, price): print("Value : ", self.valorisation(price), "Position : ", self.position(price))
    def get_portfolio_distribution(self):
        return {
            "asset":max(0, self.asset),
            "fiat":max(0, self.fiat),
            "borrowed_asset":max(0, -self.asset),
            "borrowed_fiat":max(0, -self.fiat),
            "interest_asset":self.interest_asset,
            "interest_fiat":self.interest_fiat,
        }

class TargetPortfolio(Portfolio):
    def __init__(self, position ,value, price):
        super().__init__(
            asset = position * value / price,
            fiat = (1-position) * value,
            interest_asset = 0,
            interest_fiat = 0
        )

class History:
    def __init__(self, max_size = 10000):
        self.height = max_size
    def set(self, **kwargs):
        # BUG #13 FIX: Using separate storage for numerical and object data.
        # This allows float64 optimization for reward functions while keeping flexibility.
        self.columns = []
        self.num_columns = []
        self.obj_columns = []
        
        num_values = []
        obj_values = []
        
        for name, value in kwargs.items():
            if isinstance(value, (int, float, np.number, bool)):
                self.num_columns.append(name)
                self.columns.append(name)
                num_values.append(float(value))
            else:
                self.obj_columns.append(name)
                self.columns.append(name)
                obj_values.append(value)
        
        self.num_storage = np.zeros((self.height, len(self.num_columns)), dtype=np.float64)
        self.obj_storage = np.zeros((self.height, len(self.obj_columns)), dtype='O')
        
        self.size = 0
        self.add(**kwargs)

    def add(self, **kwargs):
        num_idx = 0
        obj_idx = 0
        for name in self.columns:
            if name not in kwargs:
                # If a key is missing (e.g. reward_raw_agent), use a default
                val = 0.0 if name in self.num_columns else None
            else:
                val = kwargs[name]
                
            if name in self.num_columns:
                self.num_storage[self.size, num_idx] = float(val)
                num_idx += 1
            else:
                self.obj_storage[self.size, obj_idx] = val
                obj_idx += 1
        
        self.size = min(self.size + 1, self.height)

    def __len__(self):
        return self.size

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            column, t = arg
            if column in self.num_columns:
                col_idx = self.num_columns.index(column)
                return self.num_storage[:self.size][t, col_idx]
            else:
                col_idx = self.obj_columns.index(column)
                return self.obj_storage[:self.size][t, col_idx]
        
        if isinstance(arg, int):
            t = arg
            res = {}
            for i, name in enumerate(self.num_columns):
                res[name] = self.num_storage[:self.size][t, i]
            for i, name in enumerate(self.obj_columns):
                res[name] = self.obj_storage[:self.size][t, i]
            return res

        if isinstance(arg, str):
            column = arg
            if column in self.num_columns:
                col_idx = self.num_columns.index(column)
                return self.num_storage[:self.size][:, col_idx]
            else:
                col_idx = self.obj_columns.index(column)
                return self.obj_storage[:self.size][:, col_idx]
        
        return None

    def __setitem__(self, arg, value):
        column, t = arg
        if column in self.num_columns:
            col_idx = self.num_columns.index(column)
            self.num_storage[:self.size][t, col_idx] = float(value)
        else:
            col_idx = self.obj_columns.index(column)
            self.obj_storage[:self.size][t, col_idx] = value

def reward(history: 'History', fee_penalty: float = 2e-4,
           hold_penalty: float = 0.0, bias_penalty: float = 0.0) -> float:
    """
    Simple log-return reward with optional churn/hold/bias penalties.

    Parameters
    ----------
    history : History
        Environment history object with portfolio_valuation, position_index, etc.
    fee_penalty : float
        Penalty subtracted on each position change.
    hold_penalty : float
        Per-step penalty (encourages action; set 0 for trend-following).
    bias_penalty : float
        Penalty proportional to absolute position size.

    Returns
    -------
    float
        Scaled reward in approximately [-1, 1].
    """
    if history['idx', -1] == 0:
        return 0.0

    nav_prev, nav_now = history['portfolio_valuation', -2], history['portfolio_valuation', -1]
    
    # Absolute log return (profitability focus)
    log_return = np.log(nav_now / nav_prev)

    # Scale to be roughly [-1, 1] for typical moves (e.g. 1% move * 100 = 1.0)
    # Tanh is good for clipping extreme events
    r = np.tanh(log_return * 100)

    # Penalize churn (changing position)
    if history['position_index', -1] != history['position_index', -2]:
        r -= fee_penalty
    
    # Optional: Small penalty for holding to encourage efficiency? 
    # No, let's remove it to allow trend following.
    r -= hold_penalty
    r -= bias_penalty * abs(history['position', -1])

    # --- NEW: Volatility Penalty ---
    # Penalize large drawdowns or extreme volatility to encourage smoother equity curves
    # Calculate rolling volatility of returns (last 24 steps ~ 1 day)
    if len(history) > 24:
        returns = np.diff(np.log(history['portfolio_valuation', -25:]))
        vol = np.std(returns)
        r -= vol * 0.1 # Penalty factor

    history['reward_raw', -1] = log_return
    return r


def differential_sharpe_reward(history, eta=0.01, fee_penalty=2e-4,
                               hold_penalty=0.0, demean_market=False):
    """Differential Sharpe Ratio reward (Moody & Saffell 2001).

    Optimises the online Sharpe ratio directly by computing its gradient with
    respect to the adaptation rate η at each step.

        DSR_t = (B_{t-1}·ΔA_t - 0.5·A_{t-1}·ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

    where A and B are exponential moving averages of returns and squared returns,
    updated each step:
        A_t = A_{t-1} + η·(R_t - A_{t-1})
        B_t = B_{t-1} + η·(R_t² - B_{t-1})

    The EMA state (A, B) is stored in History columns `sharpe_A` / `sharpe_B`
    so the update is O(1) per step with no recomputation over the full window.

    If `demean_market=True`, the underlying return stream becomes
    R_t = R_agent - R_market before the Sharpe update, which removes passive
    market beta from the reward and turns buy-and-hold into ~zero alpha.
    """
    if history['idx', -1] == 0:
        return 0.0

    nav_prev = history['portfolio_valuation', -2]
    nav_now  = history['portfolio_valuation', -1]
    agent_return = _safe_log_return(nav_now, nav_prev)

    market_return = 0.0
    if demean_market:
        try:
            market_return = _market_log_return(history)
        except (KeyError, IndexError, ValueError):
            market_return = 0.0

    # Active return removes passive beta from the reward stream so the critic
    # only assigns positive advantage to state-dependent alpha.
    R_t = agent_return - market_return

    # Read previous EMA state (stored one step ago)
    A_prev = float(history['sharpe_A', -2])
    B_prev = float(history['sharpe_B', -2])

    delta_A = R_t - A_prev
    delta_B = R_t ** 2 - B_prev

    # Update EMA state for this step
    A_t = A_prev + eta * delta_A
    B_t = B_prev + eta * delta_B
    history['sharpe_A', -1] = A_t
    history['sharpe_B', -1] = B_t

    # Compute DSR
    variance = B_prev - A_prev ** 2
    if variance <= 1e-10:
        # Not enough variance yet — fall back to scaled raw return
        r = float(np.tanh(R_t * 100))
    else:
        dsr = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (variance ** 1.5)
        r = float(np.tanh(dsr * 0.01))  # bounded per-step reward; VecNormalize handles scaling

    # Transaction cost penalty
    if history['position_index', -1] != history['position_index', -2]:
        r -= fee_penalty

    r -= hold_penalty

    # NOTE: Volatility penalty removed from DSR — the Sharpe denominator
    # (B_{t-1} - A_{t-1}^2)^{3/2} already penalizes variance intrinsically.
    # Adding an explicit vol penalty double-counts and makes the agent too conservative.

    history['reward_raw', -1] = R_t
    if 'reward_raw_agent' in history.columns:
        history['reward_raw_agent', -1] = agent_return
    if 'reward_raw_market' in history.columns:
        history['reward_raw_market', -1] = market_return
    if 'reward_raw_active' in history.columns:
        history['reward_raw_active', -1] = R_t
    return r


def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def dynamic_feature_sharpe_A(history):
    """Expose DSR EMA mean (A) so the value function can condition on reward state."""
    return history['sharpe_A', -1]

def dynamic_feature_sharpe_B(history):
    """Expose DSR EMA variance (B) so the value function can condition on reward state."""
    return history['sharpe_B', -1]

def _bucketize(a, positions=None):
    """
    Map exposure in [-1,1] to bucket indices.
    BUG #6 FIX: Use actual positions to determine thresholds instead of hardcoded 0.33.
    """
    arr = np.asarray(a, dtype=float)
    if positions is not None and len(positions) >= 3:
        # Sort positions to find midpoints
        sorted_pos = np.sort(positions)
        # thresholds between buckets
        t1 = (sorted_pos[0] + sorted_pos[1]) / 2
        t2 = (sorted_pos[1] + sorted_pos[2]) / 2
        disc = np.where(arr <= t1, 0,
               np.where(arr >= t2, 2, 1)).astype(int)
    else:
        # Fallback or binary [0, 1]
        disc = np.where(arr <= 0.5, 0, 1).astype(int)
    return disc



class TradingEnv(gym.Env):
    """
    An easy trading environment for OpenAI gym. It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('TradingEnv', ...)


    :param df: The market DataFrame. It must contain 'open', 'high', 'low', 'close'. Index must be DatetimeIndex. Your desired inputs need to contain 'feature' in their column name : this way, they will be returned as observation at each step.
    :type df: pandas.DataFrame

    :param positions: List of the positions allowed by the environment.
    :type positions: optional - list[int or float]

    :param dynamic_feature_functions: The list of the dynamic features functions. By default, two dynamic features are added :
    
        * the last position taken by the agent.
        * the real position of the portfolio (that varies according to the price fluctuations)

    :type dynamic_feature_functions: optional - list   

    :param reward_function: Take the History object of the environment and must return a float.
    :type reward_function: optional - function<History->float>

    :param windows: Default is None. If it is set to an int: N, every step observation will return the past N observations. It is recommended for Recurrent Neural Network based Agents.
    :type windows: optional - None or int

    :param trading_fees: Transaction trading fees (buy and sell operations). eg: 0.01 corresponds to 1% fees
    :type trading_fees: optional - float

    :param borrow_interest_rate: Borrow interest rate per step (only when position < 0 or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP ; if your know that your borrow interest rate is 0.05% per day and that your timestep is 1 hour, you need to divide it by 24 -> 0.05/100/24.
    :type borrow_interest_rate: optional - float

    :param portfolio_initial_value: Initial valuation of the portfolio.
    :type portfolio_initial_value: float or int

    :param initial_position: You can specify the initial position of the environment or set it to 'random'. It must contained in the list parameter 'positions'.
    :type initial_position: optional - float or int

    :param max_episode_duration: If a integer value is used, each episode will be truncated after reaching the desired max duration in steps (by returning `truncated` as `True`). When using a max duration, each episode will start at a random starting point.
    :type max_episode_duration: optional - int or 'max'

    :param verbose: If 0, no log is outputted. If 1, the env send episode result logs.
    :type verbose: optional - int
    
    :param name: The name of the environment (eg. 'BTC/USDT')
    :type name: optional - str
    
    """
    metadata = {'render_modes': ['logs']}
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [-1, 0, 1],
                dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
                # BUG #9 FIX: Removed sharpe_A/B from default observations.
                # The agent shouldn't condition its strategy on its own reward denominator to avoid manipulation.
                reward_function = reward,
                windows = None,
                trading_fees = 0,
                fee_rate = 5e-4,
                slippage_bps = 2,
                slippage_order_usd = 1000.0,  # Reference notional for slippage_bps_feature
                log_frequency = 10,  # Añadir este parámetro
                borrow_interest_rate = 0,
                portfolio_initial_value = 1000,
                initial_position =0,
                max_episode_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs"
                ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.slippage_order_usd = slippage_order_usd
        self._last_trade_costs = 0.0

        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentionned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_df(df)
        
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        BIG = np.finfo(np.float64).max
        low  = np.full((self._nb_features,), -BIG, dtype=np.float64)
        high = np.full((self._nb_features,),  BIG, dtype=np.float64)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)


        if self.windows is not None:
            low  = np.full((self.windows, self._nb_features), -BIG, dtype=np.float64)
            high = np.full((self.windows, self._nb_features),  BIG, dtype=np.float64)
            self.observation_space = spaces.Box(low, high, dtype=np.float64)
        
        self.log_metrics = []
        self.log_frequency = log_frequency
        self.episode_count = 0



    def _trade(self, target_pos, price=None):
        if price is None:
            price = self._get_price()

        delta = np.clip(target_pos - self._position, -2.0, 2.0)

        new_pos = self._position + delta
        notional = abs(delta) * self._portfolio.valorisation(price)
        # Dynamic slippage: use bar-level estimate if available, else fall back to fixed
        if "slippage_bps_feature" in self.df.columns:
            # BUG #10 FIX: Rescale impact if notional differs from reference order_usd.
            # impact(Q) = impact(Q_ref) * sqrt(Q / Q_ref)
            base_slip = float(self.df.iloc[self._idx]["slippage_bps_feature"])
            spread = 1.0 # assume 1.0 bps spread constant as per add_slippage_feature
            impact_base = max(0, base_slip - spread)
            
            q_ratio = notional / max(1.0, self.slippage_order_usd)
            dyn_slip_bps = spread + impact_base * np.sqrt(q_ratio)
        else:
            dyn_slip_bps = self.slippage_bps
            
        costs = notional * (self.fee_rate + dyn_slip_bps / 1e4)
        # Use Portfolio.trade_to_position with trading_fees=0 — all costs are
        # accounted for via the explicit fee_rate + slippage model above.
        self._portfolio.trade_to_position(new_pos, price, 0.0)

        self._position = new_pos
        self._portfolio.fiat -= costs
        self._last_trade_costs = costs

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float64)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])


    
    def _get_ticker(self, delta = 0):
        return self.df.iloc[self._idx + delta]
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        if self.windows is None:
            _step_index = self._idx
        else: 
            _step_index = np.arange(self._idx + 1 - self.windows , self._idx + 1)
        return self._obs_array[_step_index]

    
    def reset(self, seed = None, options=None, **kwargs):
        super().reset(seed = seed, options = options, **kwargs)

        self._step = 0
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
        self._limit_orders = {}


        # BUG #1 FIX: Support forced start index from wrappers (RegimeBalancedWrapper)
        if options is not None and "force_start_idx" in options:
            self._idx = options["force_start_idx"]
        elif isinstance(self.max_episode_duration, int):
            min_start = (self.windows - 1) if self.windows is not None else 0
            max_start = len(self.df) - self.max_episode_duration - 1
            if max_start <= min_start:
                self._idx = min_start
            else:
                self._idx = np.random.randint(low=min_start, high=max_start)
        else:
            self._idx = 0
            if self.windows is not None:
                self._idx = self.windows - 1

        # BUG #4 FIX: Reset dynamic features in _obs_array to 0.
        # This prevents "ghost signals" from previous episodes when using windows.
        self._obs_array[:, self._nb_static_features:] = 0


        self._episode_start_idx = self._idx
       
        
        self._portfolio  = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price()
        )
        
        self.historical_info = History(max_size= len(self.df))
        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =self.positions.index(self._position),
            position = self._position,
            real_position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            **self._portfolio.get_portfolio_distribution(), # BUG #13: Flatten for numeric storage
            reward = 0.0,
            reward_raw = 0.0,
            reward_raw_agent = 0.0,
            reward_raw_market = 0.0,
            reward_raw_active = 0.0,
            sharpe_A = getattr(self.reward_function, 'dsr_a_init', 0.0),
            sharpe_B = getattr(self.reward_function, 'dsr_b_init', 0.0),
        )

        self.episode_start = True
        return self._get_obs(), self.historical_info[0]


    def render(self):
        pass


    def _take_action(self, position_index):
        target = self.positions[position_index]
        if target != self._position:
            self._trade(target)
    
    def _take_action_order_limit(self):
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                if position != self._position and params['limit'] <= ticker["high"] and params['limit'] >= ticker["low"]:
                    self._trade(position, price= params['limit'])
                    if not params['persistent']: del self._limit_orders[position]


    
    def add_limit_order(self, position, limit, persistent = False):
        self._limit_orders[position] = {
            'limit' : limit,
            'persistent': persistent
        }
    

          


    def step(self, raw_action):
        # 1️⃣ bound and apply the action
        # SB3 passes actions as length-1 arrays; NumPy 2.x refuses float()
        # on ndim>0 arrays, so squeeze to a scalar first.
        target_pos = float(np.clip(np.asarray(raw_action).reshape(-1)[0], -1, 1))
        
        # FIX: Use current price for previous valuation, not price at -1 (which wraps at idx=0)
        prev_val   = self._portfolio.valorisation(self._get_price())
        self._trade(target_pos)
        # 2️⃣ advance time
        self._idx  += 1
        self._step += 1
        self._take_action_order_limit()

        price      = self._get_price()
        self._portfolio.update_interest(self.borrow_interest_rate)
        port_val   = self._portfolio.valorisation(price)

        # 3️⃣ **derive** a discrete label only for logs / stats
        # BUG #6 FIX: Pass positions to bucketize
        position_index = _bucketize(target_pos, self.positions)

        action_str     = {0: "SHORT", 1: "HOLD", 2: "LONG"}[position_index]
        delta          = port_val - prev_val
        # print(f"[{self.df.index[self._idx]}] {action_str} …")

        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        # 4️⃣ done / truncated logic
        done       = port_val <= 0
        truncated  = (self._idx >= len(self.df)-1) or (
                    isinstance(self.max_episode_duration, int) and
                    self._step >= self.max_episode_duration-1)

        # 5️⃣ log to History
        # Around line 480 in the step() method:
        self.historical_info.add(
            idx                 = self._idx,
            step                = self._step,
            date                = self.df.index.values[self._idx],
            position_index      = position_index,
            position            = self._position,
            real_position       = self._portfolio.real_position(price),
            data                = dict(zip(self._info_columns,
                                        self._info_array[self._idx])),
            portfolio_valuation = port_val,
            **portfolio_distribution, # BUG #13: Flatten
            reward              = 0.0,
            reward_raw          = 0.0,
            reward_raw_agent    = 0.0,
            reward_raw_market   = 0.0,
            reward_raw_active   = 0.0,
            sharpe_A            = 0.0,
            sharpe_B            = 0.0,
        )

        # 6️⃣ compute reward exactly as before
        # After calculating the reward
        if not done:
            reward = self.reward_function(self.historical_info)
            
            # FIX: Remove double penalty. The fee is already reflected in portfolio_valuation change.
            # cost_penalty = self._last_trade_costs / self.portfolio_initial_value
            # reward -= cost_penalty
            
            info_fees = self._last_trade_costs   # para logs/debug
            self.historical_info["reward", -1] = reward
            
            # Also make sure to update reward_raw
            if "reward_raw" in self.historical_info.columns:
                # If there's a raw reward in the info, use it
                raw_reward = self.historical_info['reward_raw', -1]

                self.historical_info["reward_raw", -1] = raw_reward

        if done or truncated:
            # Terminal bonus scaled to be a mild shaping signal (~1-5% of typical
            # per-step DSR values), not a dominant sparse reward that overrides
            # the online Sharpe optimization.
            ep_pnl          = port_val / self.portfolio_initial_value - 1
            terminal_bonus  = 0.01 * np.tanh(2.0 * ep_pnl)
            self.historical_info["reward", -1] += terminal_bonus
            self.calculate_metrics()
            self.log()
        returned_reward_value = self.historical_info["reward", -1]
        self.episode_start = done or truncated
        return self._get_obs(), returned_reward_value, done, truncated, self.historical_info[-1]



    

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
    def calculate_metrics(self):
        # Use episode start/end prices, not the full dataset
        episode_start_price = self._price_array[self._episode_start_idx]
        episode_end_price = self._price_array[self._idx]
        market_return_value = 100 * (episode_end_price / episode_start_price - 1)
        portfolio_return_value = 100 * (self.historical_info['portfolio_valuation', -1] /
                                        self.historical_info['portfolio_valuation', 0] - 1)

        self.results_metrics = {
            "Market Return" : f"{market_return_value:.2f}%",
            "Portfolio Return" : f"{portfolio_return_value:.2f}%",
            "Market Return Value": market_return_value,
            "Portfolio Return Value": portfolio_return_value
        }
    def get_metrics(self):
        return self.results_metrics
    def log(self):
        if self.verbose > 0 and self.episode_count % self.log_frequency == 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)
        self.episode_count += 1

    def save_for_render(self, dir = "render_logs"):
        assert "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df, "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns]))
        history_df = pd.DataFrame(
            self.historical_info[columns], columns= columns
        )
        history_df.set_index("date", inplace= True)
        history_df.sort_index(inplace = True)
        render_df = self.df.join(history_df, how = "inner")
        
        if not os.path.exists(dir):os.makedirs(dir)
        render_df.to_pickle(f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")
