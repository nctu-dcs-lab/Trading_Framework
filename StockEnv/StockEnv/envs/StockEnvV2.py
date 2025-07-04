import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import os
import math
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from abc import ABC,abstractmethod



# Base class for the stock market environment
class StockMarketBase(gym.Env, ABC):
    """
    Base class for the stock market trading environment.

    This class sets up a trading environment using OpenAI Gym, allowing reinforcement learning agents
    to interact with a simulated stock market. It includes functionality for trade execution, state 
    management, normalization, and performance evaluation.

    Attributes:
        CONTRACT_PRICE (float): The fixed price per contract.
        REWARD_SCALING (float): Scaling factor for the reward.
        TRANSACTION_FEE_PERCENT (float): Transaction fee as a percentage of the trade.
        POINT_VALUE (float): The point value used for reward calculations.
        RISK_FREE (float): Risk-free return rate used in performance evaluation.
        MAX_SHARES (int): Maximum number of shares that can be owned.

    Methods:
        setup_spaces(): Sets up the action and observation spaces.
        find_min_max(): Computes min and max values for data normalization.
        load_entire_data(): Loads market data for normalization purposes.
        normalize(value, min_val, max_val): Normalizes a value between 0 and 1 based on provided min and max.
        reset_environment(): Resets the month and year for simulation.
        reset(seed=None, options=None): Resets the environment to its initial state and returns the initial observation.
        get_observation(): Returns the current normalized observation space.
        step(action): Takes a step in the environment based on the provided action (to be implemented in derived class).
        process_trade(action): Executes a trade based on the action (to be implemented in derived class).
        calculate_asset(): Calculates the total asset value (to be implemented in derived class).
        finalize_episode(asset): Finalizes the episode when done, computes the reward, and updates balance and trades.
        record_info(asset, action, reward): Records the current state information in the DataFrame.
        save_results(): Saves the simulation results to the specified file system location.
        get_info(): Provides information about the current state of the environment.
        get_performance(): Evaluates the performance of the agent in the environment.
        render(mode='human'): Renders the environment (optional, can be expanded).
    """
    
    

    def __init__(self, save_dir, train_start, train_end, start_year, end_year, init_balance=1e6) -> None:
        """
        Initializes the stock market environment with the given parameters.

        Args:
            save_dir (str): Directory path for saving results.
            train_start (int): Start year for training data.
            train_end (int): End year for training data.
            start_year (int): Initial year for simulation.
            end_year (int): Final year for simulation.
            init_balance (float): Initial balance for the agent (default is 1,000,000).
        """
        
        super().__init__()
        # Constant values used in the environment
        self.CONTRACT_PRICE = 46000
        self.REWARD_SCALING = 1e-4
        self.TRANSACTION_FEE_PERCENT = 0.00002
        self.POINT_VALUE = 50
        self.RISK_FREE = 1.7
        self.MAX_SHARES = 100
        self.WINDOW_SIZE = 76       
        # Initialize environment parameters
        self.save_dir = save_dir
        self.start_year = start_year
        self.end_year = end_year
        self.train_start = train_start
        self.train_end = train_end
        self.scalar = 15
        self.setup_spaces() # Setup action and observation spaces
        
        
        # Initialize state variables
        self.position = 'None' # Position can be 'long', 'short', or 'None'
        self.win = 0 # Number of successful trades
        self.totalRound = 0 # Total number of trades
        
        self.curr_year = None
        self.curr_month = None
        self.index = None # Current index in the DataFrame
        self.df = None # DataFrame containing market data
        self.balance = 1e6 # Current balance of the agent
        self.init_balance = init_balance # Initial balance
        self.shares = 0 # Number of shares owned
        self.total_reward = 0 # Total reward accumulated
        self.last_bid = [] # List of last bid prices
        self.mergin = 0
        
        # Initialize min and max values for normalization
        self.close_max = -np.inf
        self.volume_max = -np.inf
        self.rsi_max = -np.inf
        self.cci_max = -np.inf
        self.macd_max = -np.inf
        self.adx_max = -np.inf
        self.close_min = np.inf
        self.volume_min = np.inf
        self.rsi_min = np.inf
        self.cci_min = np.inf
        self.macd_min = np.inf
        self.adx_min = np.inf
        self.find_min_max() # Find min and max values for normalization
        self.print_param()
        
    def print_param(self):
        print(f'CONTRACT PRICE : {self.CONTRACT_PRICE}')
        print(f'REWARD SCALING : {self.REWARD_SCALING}')
        print(f'TRANSACTION FEE PERCENT : {self.TRANSACTION_FEE_PERCENT}')
        print(f'POINT VALUE : {self.POINT_VALUE}')
        print(f'RISK FREE : {self.RISK_FREE}')
        print(f'MAX SHARES : {self.MAX_SHARES}')
        print(f'WINDOW SIZE : {self.WINDOW_SIZE}')
        print(f'close max : {self.close_max}')
        print(f'close min : {self.close_min}')
        print(f'volume max : {self.volume_max}')
        print(f'volume min : {self.volume_min}')
        print(f'rsi max : {self.rsi_max}')
        print(f'rsi min : {self.rsi_min}')
        print(f'cci max : {self.cci_max}')
        print(f'cci min : {self.cci_min}')
        print(f'macd max : {self.macd_max}')
        print(f'macd min : {self.macd_min}')
        print(f'adx max : {self.adx_max}')
        print(f'adx min : {self.adx_min}')
        print(f'train start : {self.train_start}')
        print(f'train end : {self.train_end}')
        
    def setup_spaces(self):
        # Define action space (continuous) and observation space (7-dimensional)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5*self.WINDOW_SIZE+2,))
        
    def find_min_max(self):
        # Load entire dataset to find min and max for normalization
        whole_df = self.load_entire_data()
        self.close_max = np.max(whole_df['close'])
        self.volume_max = np.max(whole_df['volume'])
        self.rsi_max = np.max(whole_df['rsi'])
        self.cci_max = np.max(whole_df['cci'])
        self.macd_max = np.max(whole_df['macd'])
        self.adx_max = np.max(whole_df['adx'])
        self.close_min = np.min(whole_df['close'])
        self.volume_min = np.min(whole_df['volume'])
        self.rsi_min = np.min(whole_df['rsi'])
        self.cci_min = np.min(whole_df['cci'])
        self.macd_min = np.min(whole_df['macd'])
        self.adx_min = np.min(whole_df['adx'])

    def load_entire_data(self):
        """
        Loads the entire training dataset for normalization.

        Returns:
            pd.DataFrame: Concatenated DataFrame of the entire dataset for the specified date range.
        """
        whole_df = pd.DataFrame()
        for year in range(self.train_start, self.train_end + 1):
            for month in range(1, 13):
                file_path = f'processed_data/tfe-mtx00-{year}{month:02d}-{self.scalar}min.csv'
                temp_df = pd.read_csv(file_path)
                whole_df = pd.concat([whole_df, temp_df], axis=0, ignore_index=True)
        return whole_df    
    
    def normalize(self, value, min_val, max_val):
        """
        Normalizes a value between 0 and 1 based on provided min and max values.

        Args:
            value (float): The value to be normalized.
            min_val (float): The minimum value for normalization.
            max_val (float): The maximum value for normalization.

        Returns:
            float: Normalized value between 0 and 1.
        """        

        # Normalize a value between 0 and 1 based on min and max
        return (value - min_val) / (max_val - min_val)

    def reset_environment(self):
        """Resets month and year for simulation."""
        if self.curr_month is None or self.curr_year is None:
            self.curr_year = self.start_year
            self.curr_month = 1
        else:
            self.curr_month += 1
            if self.curr_month > 12:
                self.curr_month = 1
                self.curr_year += 1
                if self.curr_year > self.end_year:
                    self.curr_year = self.start_year
                    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generator (default is None).
            options (dict, optional): Additional options for resetting the environment (default is None).

        Returns:
            np.ndarray: An array containing the initial observation of the environment.
        """
        self.reset_environment()
        file_path = f'processed_data/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv'
        self.df = pd.read_csv(file_path)
        self.df['balance'] = self.init_balance
        self.df['asset'] = self.init_balance
        self.df['shares'] = 0.0
        self.df['action'] = 0.0
        self.df['reward'] = 0.0
        
        self.position = 'None'
        self.index = self.WINDOW_SIZE-1
        self.balance = self.init_balance
        self.shares = 0
        self.total_reward = 0
        self.last_bid = []
        self.mergin = 0
        
        self.win = 0
        self.totalRound = 0

        observation = self.get_observation()
        return observation, self.get_info()
    
    def get_observation(self):
        """
        Returns the current normalized observation space.

        Returns:
            np.ndarray: Array containing normalized state variables.
        """
        observation = [
            self.balance / (self.init_balance * 3),
            self.shares / 100,
        ]
        

        for i in range(self.WINDOW_SIZE):
            index_offset = self.index - i
            if index_offset >= 0:  
                observation.extend([
                    self.normalize(self.df.iloc[index_offset]['close'], self.close_min, self.close_max),
                    self.normalize(self.df.iloc[index_offset]['macd'], self.macd_min, self.macd_max),
                    self.normalize(self.df.iloc[index_offset]['rsi'], self.rsi_min, self.rsi_max),
                    self.normalize(self.df.iloc[index_offset]['cci'], self.cci_min, self.cci_max),
                    self.normalize(self.df.iloc[index_offset]['adx'], self.adx_min, self.adx_max),
                ])
            else:
                print(f'There is a problem with the index.')

        
        return np.array(observation, dtype=np.float32)
    @abstractmethod
    def step(self, action):
        """Take a step in the environment."""
        pass
        # return observation, reward, done, False, self.get_info()

    @abstractmethod
    def process_trade(self, action):
        """Executes a trade based on the action."""
        pass

    @abstractmethod
    def calculate_asset(self):
        """Calculates the total asset value."""
        pass


    def finalize_episode(self, asset):
        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        
        self.balance += self.CONTRACT_PRICE * self.shares
        self.balance -= (self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT) * self.shares
        for i in range(len(self.last_bid)):
            self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * self.POINT_VALUE
        
        reward = (self.balance - asset) * self.REWARD_SCALING
        
        self.index += 1 
        self.record_info(self.balance, -self.shares, reward)
        self.index -= 1 
        
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        print(f'win rate : {self.win / self.totalRound}')
        # Saving the results
        self.save_results()

        return reward
    
    def record_info(self, asset, action, reward):
        """
        Records the current state information in the DataFrame.

        Args:
            asset (float): The total asset value.
            action (float): Action taken by the agent.
            reward (float): Reward received for the action.
        """
        # Record the current state information in the DataFrame
        self.df.loc[self.index-1, 'balance'] = self.balance
        self.df.loc[self.index-1, 'asset'] = asset
        self.df.loc[self.index-1, 'shares'] = self.shares
        self.df.loc[self.index-1, 'action'] = action
        self.df.loc[self.index-1, 'reward'] = reward
        
    def save_results(self):
        """Saves the simulation results to the file system."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, f'tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv')
        self.df.to_csv(save_path, index=False)

    def get_info(self):
        """Provides information about the current state."""
        return {
            'balance': self.balance,
            'shares': self.shares,
            'close': self.df.iloc[self.index]['close'],
            'macd': self.df.iloc[self.index]['macd'],
            'rsi': self.df.iloc[self.index]['rsi'],
            'cci': self.df.iloc[self.index]['cci'],
            'adx': self.df.iloc[self.index]['adx'],
            'date': self.df.iloc[self.index]['date'],
        }
        
    def get_performance(self):
        """
        Evaluates the performance of the agent in the environment.

        Returns:
            float: The return on investment (RoR).
            float: The Sharpe ratio.
            float: The maximum drawdown.
        """
        RoR = (self.df.iloc[-1]['asset'] - self.df.iloc[0]['asset']) / self.df.iloc[0]['asset']
        mean_return = self.df['return'].mean() * 19 * 21 * 4
        std_return = self.df['return'].std() * ((19 * 21 * 4) ** 0.5)
        Sharp = (mean_return - self.RISK_FREE) / std_return
        
        cumulative_return = (self.df['cumulative return'].values / 100) + 1
        peak = np.maximum.accumulate(cumulative_return)
        peak = np.where(peak == 0, np.nan, peak)
        drawdown = (cumulative_return - peak) / peak
        drawdown = np.nan_to_num(drawdown, nan=0)

        Max_drawdown = drawdown.min() * 100
        
        return RoR, Sharp, Max_drawdown
    
    def render(self, mode='human'):
        """Renders the environment (optional, can be expanded)."""
        pass

class StockMarketLongOnly(StockMarketBase):

    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        # Adjusting the size or intensity of the action
        action = int(action * self.MAX_SHARES) # action : [-100 ~ 100]
        
        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()
        
        self.process_trade(action)
        self.index += 1
        observation = self.get_observation()
        
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        if end_asset > 0 and begin_asset > 0:
            reward = (end_asset - begin_asset) * self.REWARD_SCALING
            self.total_reward += reward
        else:
            reward = -10
            print(f'asset is negative.')

        
        self.record_info(begin_asset, action, reward)

        if self.index >= len(self.df) - 1 or end_asset <= 0:
            done = True
            reward += self.finalize_episode(end_asset)
            self.total_reward += reward
            
        return observation, reward, done, False, self.get_info()
    
    def process_trade(self, action):
        """Executes a trade based on the action."""

        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            # Available number of units traded
            available_amount = self.balance // (self.CONTRACT_PRICE + transaction_fee)
            actual_action = int(min(available_amount, action))
            
            self.balance -= (self.CONTRACT_PRICE + transaction_fee) * actual_action
            self.shares += actual_action
            
            # Record the bid price
            for _ in range(actual_action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
        
        elif action < 0: # sell
            actual_action = int(min(abs(action), self.shares))
            self.balance += self.CONTRACT_PRICE * actual_action
            self.balance -= transaction_fee * actual_action 
            
            for _ in range(actual_action):
                bid_price = self.last_bid.pop(0)
                self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
                if self.df.iloc[self.index]['close'] > bid_price:
                    self.win += 1
            self.shares -= actual_action
            
        else: # hold
            pass
    

        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * self.shares
        for i in range(int(self.shares)):
            asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
        
        return asset
    
    
class StockMarketLongDiscreteMask(StockMarketBase):

    def setup_spaces(self):
        self.action_space = spaces.Discrete(201)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5*self.WINDOW_SIZE+2,))
        
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]
        
        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()
        
        self.process_trade(action)
        self.index += 1
        observation = self.get_observation()
        
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        if end_asset > 0 and begin_asset > 0:
            reward = (end_asset - begin_asset) * self.REWARD_SCALING
            self.total_reward += reward
        else:
            reward = -10
            print(f'asset is negative.')

        
        self.record_info(begin_asset, action, reward)

        if self.index >= len(self.df) - 1 or end_asset <= 0:
            done = True
            reward += self.finalize_episode(end_asset)
            self.total_reward += reward
            
        return observation, reward, done, False, self.get_info()
    
    def process_trade(self, action):
        """Executes a trade based on the action."""

        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            # Available number of units traded
            available_amount = self.balance // (self.CONTRACT_PRICE + transaction_fee)
            actual_action = int(min(available_amount, action))
            
            self.balance -= (self.CONTRACT_PRICE + transaction_fee) * actual_action
            self.shares += actual_action
            
            # Record the bid price
            for _ in range(actual_action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
        
        elif action < 0: # sell
            actual_action = int(min(abs(action), self.shares))
            self.balance += self.CONTRACT_PRICE * actual_action
            self.balance -= transaction_fee * actual_action 
            
            for _ in range(actual_action):
                bid_price = self.last_bid.pop(0)
                self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
                if self.df.iloc[self.index]['close'] > bid_price:
                    self.win += 1
            self.shares -= actual_action
            
        else: # hold
            pass
    
        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * self.shares
        for i in range(int(self.shares)):
            asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
        
        return asset


    def valid_action_mask(self):
        actions_mask = np.ones(201, dtype=int)
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))

        for i in range(100+available_amount, 201):
            actions_mask[i] = 0
        
        for i in range(0, 100-self.shares):
            actions_mask[i] = 0    
        return actions_mask.astype(bool)


class StockMarketLongDiscreteMaskV2(StockMarketBase):
    '''The reward function in this environment is different from StockMarketLongDiscreteMask.'''
    def setup_spaces(self):
        self.action_space = spaces.Discrete(201)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5*self.WINDOW_SIZE+2,))
        
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]
        
        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()
        

        # process trade
        reward = self.process_trade(action)
        self.total_reward += reward
        
        #get next obs
        self.index += 1
        observation = self.get_observation()
        
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        #record some useful info
        self.record_info(begin_asset, action, reward)

        if self.index >= len(self.df) - 1 or end_asset <= 0:
            done = True
            reward += self.finalize_episode(end_asset)
            self.total_reward += reward
            print(f'total reward : {self.total_reward}')  
            
        return observation, reward, done, False, self.get_info()
    
    def process_trade(self, action):
        """Executes a trade based on the action."""

        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            # Available number of units traded
            available_amount = self.balance // (self.CONTRACT_PRICE + transaction_fee)
            actual_action = int(min(available_amount, action))
            
            self.balance -= (self.CONTRACT_PRICE + transaction_fee) * actual_action
            self.shares += actual_action
            
            # Record the bid price
            for _ in range(actual_action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
        
        elif action < 0: # sell
            actual_action = int(min(abs(action), self.shares))
            self.balance += self.CONTRACT_PRICE * actual_action
            self.balance -= transaction_fee * actual_action 
            
            for _ in range(actual_action):
                bid_price = self.last_bid.pop(0)
                self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
                reward += self.df.iloc[self.index]['close'] - bid_price
                if self.df.iloc[self.index]['close'] > bid_price:
                    self.win += 1
            self.shares -= actual_action
            
        else: # hold
            pass
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
        return reward
        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * self.shares
        for i in range(int(self.shares)):
            asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
        
        return asset


    def valid_action_mask(self):
        actions_mask = np.ones(201, dtype=int)
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))

        for i in range(101+available_amount, 201):
            actions_mask[i] = 0
        
        for i in range(0, 100-self.shares):
            actions_mask[i] = 0    
        return actions_mask.astype(bool)  
    
    def finalize_episode(self, asset):
        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        reward = 0
        self.balance += self.CONTRACT_PRICE * self.shares
        self.balance -= (self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT) * self.shares
        for i in range(len(self.last_bid)):
            bid_price = self.last_bid.pop(0)
            self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
            reward += self.df.iloc[self.index]['close'] - bid_price
       
        
        self.index += 1 
        self.record_info(self.balance, -self.shares, reward)
        self.index -= 1 
        
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        print(f'win rate : {self.win / self.totalRound}')
        # Saving the results
        self.save_results()
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
        return reward
    
class StockMarketShortOnly(StockMarketBase):
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0
        
        # Adjusting the size or intensity of the action
        action = int(action * self.MAX_SHARES) # action : [-100 ~ 100]
        
        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()
        
        self.process_trade(action)
        self.index += 1
        observation = self.get_observation()
        
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        if end_asset > 0 and begin_asset > 0:
            reward = math.log(end_asset/begin_asset) * 100
            self.total_reward += reward
        else:
            reward = -10
            print(f'asset is negative.')
        
        # Calculate win rate
        self.totalRound += 1
        if (begin_asset < end_asset):
            self.win += 1
        
        self.record_info(begin_asset, action, reward)

        if self.index >= len(self.df) - 1 or end_asset <= 0:
            done = True
            reward += self.finalize_episode(end_asset)
            self.total_reward += reward
            
        return observation, reward, done, False, self.get_info()
    
    def process_trade(self, action):
        """Executes a trade based on the action."""
        

        
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        
        # Determine whether to buy, sell, or hold
        if action > 0 : # buy
            actual_action = int(min(action, self.shares))
            self.balance += self.CONTRACT_PRICE * actual_action
            self.balance -= transaction_fee * actual_action 
            for _ in range(actual_action):
                bid_price = self.last_bid.pop(0)
                # Multiply by -1 because it's a short position
                self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE * -1 
                if self.df.iloc[self.index]['close'] < bid_price:
                    self.win += 1
            self.shares -= actual_action
            
        elif action < 0: # sell
            # Available number of units traded
            available_amount = self.balance // (self.CONTRACT_PRICE + transaction_fee)
            actual_action = int(min(available_amount, abs(action)))
            
            self.balance -= (self.CONTRACT_PRICE + transaction_fee) * actual_action
            self.shares += actual_action
            
            # Record the bid price
            for _ in range(actual_action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
                
        else:
            pass
    
        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * self.shares
        for i in range(int(self.shares)):
            # Multiply by -1 because it's a short position
            asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i]) * -1
        
        return asset
    