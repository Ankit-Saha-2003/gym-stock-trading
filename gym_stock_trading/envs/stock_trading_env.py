import gym
from gym import spaces

import numpy as np

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(StockTradingEnv, self).__init__()

        # Maximum number of shares that can be held
        self.max_shares = 100_000

        # Number of shares held       
        discrete_space = spaces.Discrete(self.max_shares + 1)

        # The continuous space consists of the following:
        #   Balance,
        #   Closing price,
        # Technical Indicators:
        #   RSI,
        #   Simple Moving Average, 
        #   Exponential Moving Average,
        #   Stochastic Oscillator,
        #   MACD,
        #   Accumulation/Distribution Oscillator,
        #   On-Balance Volume (OBV),
        #   Price Rate of Change (ROC),
        #   William's %R,
        #   Disparity Index
        continuous_space = spaces.Box(low=-1e5, high=1e5, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Tuple((discrete_space, continuous_space))

        # Action space is in range [-1, 1]
        # Scale it by max_shares to obtain the number of shares to be processed
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)

        # Check if the current render_mode is supported
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Current time instant
        self.timestamp = 0

    def _get_obs(self):
        # Retrieve the observation space at any step
        pass

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
