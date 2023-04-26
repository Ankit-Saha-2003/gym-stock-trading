import gym
from gym import spaces

import numpy as np

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(StockTradingEnv, self).__init__()

        # Observation space consisting of all the input parameters in the range [1, 1e6]
        # Proposed Observation Space -> no. of shares, balance, closing price, Technical Indicators(10)

        discrete_space = spaces.Discrete(1e5) # max no. of shares taken 1e5 
        continuous_space = spaces.Box(low=0,high =1e6,dtype = np.float32)

        self.observation_space = spaces.Tuple((discrete_space,continuous_space))

        # self.observation_space = spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32)

        # Action space consisting of two integers: 
        # one for hold/buy/sell, and the other for the number of stocks on which the action has to be taken
        self.action_space = spaces.MultiDiscrete([3, 1_000_000])

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
