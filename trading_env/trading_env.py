import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime
import glob
from pathlib import Path    

from collections import Counter

import tempfile, os
import warnings
warnings.filterwarnings("error")

class Portfolio:
    def __init__(self, asset, fiat, interest_asset = 0, interest_fiat = 0):
        self.asset =asset
        self.fiat =fiat
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
        self.interest_asset = max(0, - self.asset)*borrow_interest_rate
        self.interest_fiat = max(0, - self.fiat)*borrow_interest_rate
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
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)
        
        self.width = len(self.columns)
        self.history_storage = np.zeros((self.height, self.width), dtype='O')
        
        self.size = 0
        self.add(**kwargs)
    def add(self, **kwargs):
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.columns:
            self.history_storage[self.size, :] = values
            self.size = min(self.size+1, self.height)
        else:
            raise ValueError(f"Make sur that your inputs match the initial ones... Initial ones : {self.columns}. New ones {columns}")
    def __len__(self):
        return self.size
    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            column, t = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][t, column_index]
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[:self.size][t]))
        if isinstance(arg, str):
            column = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_index]
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                try:
                    column_indexes.append(self.columns.index(column))
                except ValueError as e:
                    raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_indexes]

    def __setitem__(self, arg, value):
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError as e:
            raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
        self.history_storage[:self.size][t, column_index] = value


import numpy as np



# trading_env.py
def reward(history, fee_penalty: float = 2e-4):
    # ► salimos en el primer tick del episodio
    if history['idx', -1] == 0:
        return 0.0

    nav_prev, nav_now = history['portfolio_valuation', -2], history['portfolio_valuation', -1]
    px_prev, px_now   = history['data_close', -2],         history['data_close', -1]

    pnl = np.log(nav_now / nav_prev)
    mkt = np.log(px_now  / px_prev)

    excess = pnl - mkt
    reward = np.tanh(150 * excess)          # rango aprox. ±1

    if history['position_index', -1] != history['position_index', -2]:
        reward -= fee_penalty                 # castigo por rotar

    history['reward_raw', -1] = excess        # tracking
    return np.tanh(150 * excess)  # rango aprox. ±1



def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def _bucketize(a):
    """
    Map exposure in [-1,1] to bucket 0=SHORT, 1=HOLD, 2=LONG.
    Works for scalars or arrays; always returns a NumPy array.
    """
    arr = np.asarray(a, dtype=float)
    disc = np.where(arr <= -0.33, -1,
            np.where(arr >= 0.33,  1, 0)).astype(int) + 1   # → {0,1,2}
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
                reward_function = reward,
                windows = None,
                trading_fees = 0,
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



    def _trade(self, target_pos, price=None):
        if price is None:
            price = self._get_price()

       
        
        delta = np.clip(target_pos - self._position, -1.0, 1.0)

        new_pos = self._position + delta
        self._portfolio.trade_to_position(new_pos, price, self.trading_fees)
        self._position = new_pos

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
        

        if isinstance(self.max_episode_duration, int):
            max_start = len(self.df) - self.max_episode_duration - 1
            self._idx = np.random.randint(low=0, high=max_start)
        else:
            self._idx = 0
        if self.windows is not None: self._idx = self.windows - 1
       
        
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
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,
            reward_raw = 0,
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
        target_pos = float(np.clip(raw_action, -1, 1))
        

        prev_val   = self._portfolio.valorisation(self._get_price(-1))
        self._trade(target_pos)
        # 2️⃣ advance time
        self._idx  += 1
        self._step += 1
        self._take_action_order_limit()

        price      = self._get_price()
        self._portfolio.update_interest(self.borrow_interest_rate)
        port_val   = self._portfolio.valorisation(price)

        # 3️⃣ **derive** a discrete label only for logs / stats
        position_index = _bucketize(target_pos)        # ← NEW
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
            portfolio_distribution = portfolio_distribution,
            reward              = 0.0,
            reward_raw          = 0.0  # Add this line to fix the error
        )

        # 6️⃣ compute reward exactly as before
        # After calculating the reward
        if not done:
            reward = self.reward_function(self.historical_info)
            self.historical_info["reward", -1] = reward
            
            # Also make sure to update reward_raw
            if "reward_raw" in self.historical_info.columns:
                # If there's a raw reward in the info, use it
                raw_reward = self.historical_info['reward_raw', -1]

                self.historical_info["reward_raw", -1] = raw_reward

        if done or truncated:
            ep_pnl          = port_val / self.portfolio_initial_value - 1
            terminal_bonus  = 0.5 * np.tanh(5*ep_pnl)
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
        # Usar df en lugar de historical_info para calcular el return de mercado
        market_return_value = 100 * (self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1)
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
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

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



