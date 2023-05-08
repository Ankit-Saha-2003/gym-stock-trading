import gym
from gym import spaces

import numpy as np
import pandas as pd
import Technical_Indicators as TI

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,balance,file_type,file_path,render_mode=None):
        super(StockTradingEnv, self).__init__()

        if file_type == "csv":
             self.data = pd.read_csv(file_path)
        elif file_type == "excel":
             self.data = pd.read_excel(file_path)
        else :
            raise Exception(TypeError)

        # Proposed Observation Space -> no. of shares, balance, closing price, Technical Indicators(10)

        discrete_space = spaces.Discrete(1e5) # max no. of shares taken 1e5 

        #  In continuous_space we have (at time t) ,
        #  'Balance' ,
        #  'Closing price' ,
        # Technical Indicators --- start from here----
        #   RSI  ,
        #   Simple Moving Average, Exponential Moving Average ,
        #   Stochastic Oscillator,
        #   MACD,
        #   Accumulation/Distribution Oscillator,
        #   On-Balance Volume (OBV),
        #   Price Rate of Change (ROC),
        #   William's %R,
        #   Disparity Index
             
        continuous_space = spaces.Box(low=-1e5,high =1e5,shape = (12,),dtype = np.float32)

        self.observation_space = spaces.Tuple((discrete_space,continuous_space))

        # self.observation_space = spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32)

        # New action space is in range [-1,1] and rescale it with max_no. of shares to find no.of shares to be processed
        self.max_shares = 1e5  # Temporary

        self.action_space = spaces.Box(low = -1,high = 1,dtype=np.float32)

        # Initialising some state variables
        self.no_of_shares = 0
        self.balance = balance

        # Check if the current render_mode is supported
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Current time instant
        self.timestamp = 0

    def asset_price(self):
           return self.data.iloc[self.timestamp]['Adj Close']

    def _get_obs(self,window_size = 10):
        # Retrieve the observation space at any step
        obs_state = {
                    "no of shares": self.no_of_shares,
                    "balance" : self.balance,
                    "closing price" : self.asset_price(),
                    "RSI" : TI.RSI(self.data,self.timestamp,window_size),
                    "SMA" : TI.SMA(self.data,self.timestamp,window_size),
                    "EMA" : TI.EMA(self.data,self.timestamp,window_size),
                    "Stochastic Oscillator" : TI.stochastic_oscillator(self.data,self.timestamp,window_size),
                    "MACD": TI.MACD(self.data,self.timestamp,window_size),
                    "AD": TI.AD(self.data,self.timestamp),
                    "OBV": TI.OBV(self.data,self.timestamp),
                    "ROC": TI.PROC(self.data,self.timestamp,window_size),
                    "William%R" : TI.William_R(self.data,self.timestamp,window_size),
                    "DisparityIndex" : TI.Disparity_index(self.data,self.timestamp,window_size)
        }

        return obs_state
       

    def _get_info(self):
        # Retrieve auxiliary information about the current state
        pass

    def _get_reward(self):
        # Compute the reward for performing a particular action
        pass

    def _take_action(self, action):
        # Takes an action in the current state
        
        # Hold
        if action[0] == 0:
            return

        # Buy
        elif action[0] == 1:
            if self.current_price * action[1] > self.cash_in_hand:
                raise ValueError('Insufficient money')

            self.shares_held += action[1]
            self.cash_in_hand -= self.current_price * action[1]

        # Sell
        else:
            if action[1] > self.shares_held:
                raise ValueError('Insufficient stocks')

            self.shares_held -= action[1]
            self.cash_in_hand += self.current_price * action[1]

    def step(self, action):
        # Execute one time step in the environment and return the updated state, reward, termination and auxiliar information
        
        # Check if the action is valid
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')

        self._take_action(action)
        self.timestamp += 1
        self.current_price = self.stock_prices[self.timestamp]
        self.net_worth = self.cash_in_hand + self.shares_held * self.current_price 

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = True if self.timestamp >= len(self.df) else False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def reset(self):
        # Reset the environment to its initial state
        pass

    def render(self):
        # Visualize the environment
        pass

    def close(self):
        # Close the environment
        pass
