import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, A2C, DQN

INITIAL_ACCOUNT_BALANCE = 1e4 

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(StockTradingEnv, self).__init__()


    def _get_obs(self):
        # Retrieve the observation space at any step
        pass

    def _get_info(self):
        # Retrieve auxiliary information about the current state
        pass

    def step(self, action):
        # Execute one time step in the environment and return the updated state, reward, termination and auxiliar information
        pass

    def reset(self):
        # Reset the environment to its initial state
        self.initial_capital = INITIAL_ACCOUNT_BALANCE
        self.portfolio_value = INITIAL_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.timestamp = 1                                # Random ?
        self.num_shares = 0
        self.time_list, self.reward_list, self.close_price_list = [], [], []

        return self._get_obs()


    def render(self):
        # Visualize the environment
        print(f'Step: {self.timestamp}')
        
        # Plot Reward vs Timestamp
        self.time_list.append(self.timestamp) 
        self.reward_list.append(self._get_reward())
        plt.figure(figsize=(12,6))
        plt.plot(self.time_list, self.reward_list)
        plt.xlabel('Timestamp')
        plt.ylabel('Reward')
        plt.title('Reward vs Timestamp')
        plt.xticks(np.arange(len(self.data)))
        plt.show()

        # Plot Closing Price vs Timestamp
        self.close_price_list.append(self.current_price)
        plt.figure(figsize=(12,6))
        plt.plot(self.time_list, self.close_price_list)
        plt.xlabel('Timestamp')
        plt.ylabel('Closing Price')
        plt.title('Closing Price vs Timestamp')
        plt.xticks(np.arange(len(self.data)))
        plt.show()


    def train(self, model_type='PPO', model_save_path='model.zip'):
        if model_type == 'PPO':
            model = PPO('MlpPolicy', self, verbose=1)
        elif model_type == 'A2C':
            model = A2C('MlpPolicy', self, verbose=1)
        elif model_type == 'DQN':
            model = DQN('MlpPolicy', self, verbose=1)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Train the model
        num_timesteps = len(self.data)
        model.learn(total_timesteps=num_timesteps)

        # Save the trained model
        model.save(model_save_path)

        # Close the environment
        self.close()

    
    def evaluate_PPO(self):
        # Load pre-trained model
        model = PPO.load("model.zip", env=self)

        # Evaluate pre-trained model
        obs = self.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.step(action)
            self.render()
            self.close()



    def close(self):
        # Close the environment
        plt.close()  # Close all open plots
