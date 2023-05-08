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
        return {"Current net worth": self.portfolio_value}

    def _get_reward(self):
        # Compute the reward at each timestep
        new_portfolio_value = self.balance + self.num_shares * self.current_price
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value
        return reward

    def _take_action(self, action):
        # Takes an action in the current state
        
        # Hold
        if action == 0:
            return

        # Buy
        elif action > 0:
            buy_shares = action * self.num_shares 
            if self.current_price * buy_shares > self.balance:
                raise ValueError('Insufficient money')

            self.num_shares += buy_shares
            self.balance -= self.current_price * buy_shares

        # Sell
        else:
            sell_shares = -1 * action * self.num_shares
            if sell_shares > self.num_shares:
                raise ValueError('Insufficient shares')

            self.num_shares -= sell_shares
            self.balance += self.current_price * sell_shares

    def step(self, action):
        # Execute one time step in the environment and return the updated state, reward, termination and auxiliar information
        
        # Check if the action is valid
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
            
        self.current_price = "Call function here"
        self._take_action(action)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = True if self.timestamp >= len(self.df) else False
        truncated = False
        info = self._get_info()

        self.timestamp += 1
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
