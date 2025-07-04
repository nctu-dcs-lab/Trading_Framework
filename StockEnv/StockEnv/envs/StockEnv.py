import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import os
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import time
# 100 shares per trade
MAX_SHARES = 100


TRANSACTION_FEE_PERCENT = 0.00002
REWARD_SCALING = 1e-4
CONTRACT_PRICE = 46000
POINT_VALUE = 50
RISK_FREE = 1.7

class StockMarket(gym.Env):
    
    def __init__(self, save_dir, train_start, train_end, start_year, end_year) -> None:
        super().__init__()
        
        self.start_year = start_year
        self.end_year = end_year
        self.train_start = train_start
        self.train_end = train_end
        print(f'range of environmental training : {self.train_start} to {self.train_end}')
        print(f'range of environmental operation : {self.start_year} to {self.end_year}')
        self.scalar = 15
        
        self.save_dir = save_dir
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,)) #O H L C V 
        
        self.curr_year = None
        self.curr_month = None
        self.index = None
        self.df = None
        self.balance = 1e6
        self.shares = 0
        self.total_reward = 0
        
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
        self.find_min_max()


    
    def find_min_max(self):
        for year in range(self.train_start, self.train_end+1):
            for month in range(1,13):
                if year == self.train_start and month == 1:
                    whole_df = pd.read_csv(f'processed_data/tfe-mtx00-{year}{month:02d}-{self.scalar}min.csv')
                else:
                    temp_df = pd.read_csv(f'processed_data/tfe-mtx00-{year}{month:02d}-{self.scalar}min.csv')
                    whole_df = pd.concat([whole_df, temp_df], axis=0, ignore_index=True)
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

    def get_info(self):
        info = []
        if self.df is None:
            print(f'please reset env first.')
            return None
        else:

            info.append(int(self.balance))
            info.append(int(self.shares))
            info.append(self.df.iloc[self.index]['close'])
            info.append(self.df.iloc[self.index]['macd'])
            info.append(self.df.iloc[self.index]['rsi'])
            info.append(self.df.iloc[self.index]['cci'])
            info.append(self.df.iloc[self.index]['adx'])
            info.append(self.df.iloc[self.index]['date'])
            
        return {'balance':self.balance, 
                'shares':self.shares, 
                'close':self.df.iloc[self.index]['close'], 
                'macd':self.df.iloc[self.index]['macd'],
                'rsi':self.df.iloc[self.index]['rsi'],
                'cci':self.df.iloc[self.index]['cci'],
                'adx':self.df.iloc[self.index]['adx'],
                'date':self.df.iloc[self.index]['date'],}

    def get_performance(self):
        RoR = (self.df.iloc[-1]['asset'] - self.df.iloc[0]['asset']) / self.df.iloc[0]['asset']
        mean_return = self.df['return'].mean() * 19 * 21 * 4
        std_return = self.df['return'].std() * ((19 * 21 * 4) ** 0.5)
        Sharp = (mean_return - RISK_FREE) / std_return
        
        cumulative_return = (self.df['cumulative return'].values / 100) + 1
        peak = np.maximum.accumulate(cumulative_return)
        peak = np.where(peak == 0, np.nan, peak)
        drawdown = (cumulative_return - peak) / peak
        drawdown = np.nan_to_num(drawdown, nan=0)

        Max_drawdown = drawdown.min() * 100
        
        return RoR, Sharp, Max_drawdown
            
    def get_obs(self):
        observation = []
        if self.df is None:
            print(f'please reset env first.')
            return None
        else:

            # Normalize the data using min-max normalization
            observation.append((self.balance) / (1e6 * 3))
            observation.append((self.shares) / (100))
            observation.append((self.df.iloc[self.index]['close'] - self.close_min) / (self.close_max - self.close_min))
            observation.append((self.df.iloc[self.index]['macd'] - self.macd_min) / (self.macd_max - self.macd_min))
            observation.append((self.df.iloc[self.index]['rsi'] - self.rsi_min) / (self.rsi_max - self.rsi_min))
            observation.append((self.df.iloc[self.index]['cci'] - self.cci_min) / (self.cci_max - self.cci_min))
            observation.append((self.df.iloc[self.index]['adx'] - self.adx_min) / (self.adx_max - self.adx_min))
        return np.array(observation, dtype=np.float32)
            
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Returns:
            np.ndarray: The initial observation of the environment.
        """

        if self.curr_month == None or self.curr_year == None:
            self.curr_year = self.start_year
            self.curr_month = 1
        
        else:
            self.curr_month += 1
            if self.curr_month >= 13:
                self.curr_month = 1
                self.curr_year += 1
                if self.curr_year >= self.end_year + 1:
                    self.curr_year = self.start_year
       
        self.df = pd.read_csv(f'processed_data/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv')
        self.df.insert(len(self.df.columns), 'balance', 0.0)
        self.df.insert(len(self.df.columns), 'asset', 0.0)
        self.df.insert(len(self.df.columns), 'shares', 0.0)
        self.df.insert(len(self.df.columns), 'action', 0.0)
        self.index = 0
        self.balance = 1e6
        self.shares = 0
        self.total_reward = 0
        self.last_bid = []
        
        observartion = self.get_obs()
        return (observartion, self.get_info())
    
    def step(self, action):
        reward = 0
        done = False


        if self.index >= len(self.df) - 1:
            observartion = self.get_obs()
            done = True
            
            self.balance += CONTRACT_PRICE * self.shares
            self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * self.shares
            for i in range(int(self.shares)):
                self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE
            
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'shares'] = self.shares
            self.df.loc[self.index, 'action'] = -self.shares
            self.df.loc[self.index, 'asset'] = self.balance
            self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
            self.df['cumulative return'] = ((1 + (self.df['return'] / 100)).cumprod() - 1) * 100

            performance = self.get_performance()
            print(f'date : {self.curr_year}/{self.curr_month}, balance : {self.balance}, total_reward : {self.total_reward}, RoR : {performance[0]}, Sharp : {performance[1]}, MDD : {performance[2]}')
            
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv')
            self.df.to_csv(save_path, index=False)
            
        else:
            action = int(action * MAX_SHARES)
            
            begin_asset = self.balance + CONTRACT_PRICE * self.shares
            for i in range(int(self.shares)):
                begin_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
                
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'asset'] = begin_asset
            self.df.loc[self.index, 'shares'] = self.shares
            
            #buy
            if action > 0:
                available_amount = self.balance // (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT)
                self.balance -= (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(available_amount, action)
                self.df.loc[self.index+1, 'action'] = int(min(available_amount, action))
                
                self.shares += min(available_amount, action)
                for i in range(int(min(available_amount, action))):
                    self.last_bid.append(self.df.iloc[self.index]['close']) 
            #sell
            elif action < 0:
                self.balance += CONTRACT_PRICE * min(abs(action), self.shares)
                self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(abs(action), self.shares)
                self.df.loc[self.index+1, 'action'] = -1 * int(min(abs(action), self.shares))
                
                for i in range(int(min(abs(action), self.shares))):
                    self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE
                self.shares -= min(abs(action), self.shares)
                    

            self.index += 1
            
            observartion = self.get_obs()
            end_asset = self.balance + CONTRACT_PRICE * self.shares
            for i in range(int(self.shares)):
                end_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])


            reward = (end_asset - begin_asset) * REWARD_SCALING
            self.total_reward += reward
        return observartion, reward, done, False, self.get_info()


    def render(self):
        pass


class StockMarketValid(StockMarket):
    def __init__(self, save_dir, start_year, end_year, train_start, train_end):
        super().__init__(save_dir=save_dir, 
                         start_year=start_year, 
                         end_year=end_year, 
                         train_start=train_start, 
                         train_end=train_end)

class StockMarketLS(StockMarket):
    def __init__(self, save_dir, start_year, end_year, train_start, train_end):
        super().__init__(save_dir=save_dir, 
                         start_year=start_year, 
                         end_year=end_year, 
                         train_start=train_start, 
                         train_end=train_end)
    def get_obs(self):
        observation = []
        if self.df is None:
            print(f'please reset env first.')
            return None
        else:

            # Normalize the data using min-max normalization
            observation.append((self.balance) / (1e6 * 3))
            observation.append((self.shares + MAX_SHARES) / (2 * MAX_SHARES))
            observation.append((self.df.iloc[self.index]['close'] - self.close_min) / (self.close_max - self.close_min))
            observation.append((self.df.iloc[self.index]['macd'] - self.macd_min) / (self.macd_max - self.macd_min))
            observation.append((self.df.iloc[self.index]['rsi'] - self.rsi_min) / (self.rsi_max - self.rsi_min))
            observation.append((self.df.iloc[self.index]['cci'] - self.cci_min) / (self.cci_max - self.cci_min))
            observation.append((self.df.iloc[self.index]['adx'] - self.adx_min) / (self.adx_max - self.adx_min))
        return np.array(observation, dtype=np.float32)
    
    def step(self, action):
        reward = 0
        done = False


        if self.index >= len(self.df) - 1:
            observartion = self.get_obs()
            done = True
            if self.shares >= 0:
                self.balance += CONTRACT_PRICE * self.shares
                self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * self.shares
                for i in range(int(self.shares)):
                    self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE
                
                self.df.loc[self.index, 'shares'] = self.shares
                self.df.loc[self.index, 'action'] = -self.shares
            else:
                self.balance += CONTRACT_PRICE * abs(self.shares)
                self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * abs(self.shares)
                for i in range(int(abs(self.shares))):
                    self.balance += (self.last_bid.pop(0) - self.df.iloc[self.index]['close']) * POINT_VALUE
                
                self.df.loc[self.index, 'shares'] = self.shares
                self.df.loc[self.index, 'action'] = abs(self.shares)
                                
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'asset'] = self.balance
            self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
            self.df['cumulative return'] = ((1 + (self.df['return'] / 100)).cumprod() - 1) * 100

            performance = self.get_performance()
            print(f'date : {self.curr_year}/{self.curr_month}, balance : {self.balance}, total_reward : {self.total_reward}, RoR : {performance[0]}, Sharp : {performance[1]}, MDD : {performance[2]}')
            
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv')
            self.df.to_csv(save_path, index=False)
                
        else:
            action = int(action * MAX_SHARES)
            
            begin_asset = self.balance + CONTRACT_PRICE * abs(self.shares)
            for i in range(int(abs(self.shares))):
                if self.shares >= 0:
                    begin_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
                else:
                    begin_asset += POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])
                
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'asset'] = begin_asset
            self.df.loc[self.index, 'shares'] = self.shares
            
            #buy
            if action > 0:
                #long
                if self.shares >= 0:
                    available_amount = self.balance // (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT)
                    self.balance -= (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(available_amount, action)
                    self.df.loc[self.index+1, 'action'] = int(min(available_amount, action))
                    
                    self.shares += int(min(available_amount, action))
                    for i in range(int(min(available_amount, action))):
                        self.last_bid.append(self.df.iloc[self.index]['close']) 
                    
                #short
                else:
                    self.balance += CONTRACT_PRICE * min(action, abs(self.shares))
                    self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(action, abs(self.shares))
                    self.df.loc[self.index+1, 'action'] = int(min(action, abs(self.shares)))
                    
                    for i in range(int(min(action, abs(self.shares)))):
                        self.balance += (self.last_bid.pop(0) - self.df.iloc[self.index]['close']) * POINT_VALUE
                    
                    self.shares += min(action, abs(self.shares))

            #sell
            elif action < 0:
                #long
                if self.shares > 0:
                    self.balance += CONTRACT_PRICE * min(abs(action), self.shares)
                    self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(abs(action), self.shares)
                    self.df.loc[self.index+1, 'action'] = -1 * int(min(abs(action), self.shares))
                    
                    for i in range(int(min(abs(action), self.shares))):
                        self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE

                    self.shares -= min(abs(action), self.shares)

                else:
                    available_amount = self.balance // (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT)
                    self.balance -= (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(available_amount, abs(action))
                    self.df.loc[self.index+1, 'action'] = -1 * int(min(available_amount, abs(action)))
                    
                    self.shares -= int(min(available_amount, abs(action)))
                    for i in range(int(min(available_amount, abs(action)))):
                        self.last_bid.append(self.df.iloc[self.index]['close'])                     
                    

            self.index += 1
            
            observartion = self.get_obs()
            end_asset = self.balance + CONTRACT_PRICE * abs(self.shares)

            for i in range(int(abs(self.shares))):
                if self.shares >= 0:
                    end_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
                else:
                    end_asset += POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])

            if end_asset <= 0:
                reward = -100
                
                self.df.loc[self.index, 'balance'] = end_asset
                self.df.loc[self.index, 'asset'] = end_asset
                self.balance = end_asset
                self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
                self.df['cumulative return'] = ((1 + (self.df['return'] / 100)).cumprod() - 1) * 100
                
                performance = self.get_performance()
                print(f'date : {self.curr_year}/{self.curr_month}, balance : {self.balance}, total_reward : {self.total_reward}, RoR : {performance[0]}, Sharp : {performance[1]}, MDD : {performance[2]}') 
                done = True
            else: 
                reward = (end_asset - begin_asset) * REWARD_SCALING
            
            self.total_reward += reward
        return observartion, reward, done, False, self.get_info()

class StockMarketSharp(StockMarket):
    def __init__(self, save_dir, start_year, end_year, train_start, train_end):
        super().__init__(save_dir=save_dir, 
                         start_year=start_year, 
                         end_year=end_year, 
                         train_start=train_start, 
                         train_end=train_end)

    def get_obs(self):
        observation = []
        if self.df is None:
            print(f'please reset env first.')
            return None
        else:

            # Normalize the data using min-max normalization
            observation.append((self.balance) / (1e6 * 3))
            observation.append((self.shares + MAX_SHARES) / (2 * MAX_SHARES))
            observation.append((self.df.iloc[self.index]['close'] - self.close_min) / (self.close_max - self.close_min))
            observation.append((self.df.iloc[self.index]['macd'] - self.macd_min) / (self.macd_max - self.macd_min))
            observation.append((self.df.iloc[self.index]['rsi'] - self.rsi_min) / (self.rsi_max - self.rsi_min))
            observation.append((self.df.iloc[self.index]['cci'] - self.cci_min) / (self.cci_max - self.cci_min))
            observation.append((self.df.iloc[self.index]['adx'] - self.adx_min) / (self.adx_max - self.adx_min))
        return np.array(observation, dtype=np.float32)
    
    def step(self, action):
        reward = 0
        done = False


        if self.index >= len(self.df) - 1:
            observartion = self.get_obs()
            done = True
            if self.shares >= 0:
                self.balance += CONTRACT_PRICE * self.shares
                self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * self.shares
                for i in range(int(self.shares)):
                    self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE
                
                self.df.loc[self.index, 'shares'] = self.shares
                self.df.loc[self.index, 'action'] = -self.shares
            else:
                self.balance += CONTRACT_PRICE * abs(self.shares)
                self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * abs(self.shares)
                for i in range(int(abs(self.shares))):
                    self.balance += (self.last_bid.pop(0) - self.df.iloc[self.index]['close']) * POINT_VALUE
                
                self.df.loc[self.index, 'shares'] = self.shares
                self.df.loc[self.index, 'action'] = abs(self.shares)
                                
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'asset'] = self.balance
            self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
            self.df['cumulative return'] = ((1 + (self.df['return'] / 100)).cumprod() - 1) * 100

            performance = self.get_performance()
            print(f'date : {self.curr_year}/{self.curr_month}, balance : {self.balance}, total_reward : {self.total_reward}, RoR : {performance[0]}, Sharp : {performance[1]}, MDD : {performance[2]}')
            
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv')
            self.df.to_csv(save_path, index=False)
                
        else:
            action = int(action * MAX_SHARES)
            
            begin_asset = self.balance + CONTRACT_PRICE * abs(self.shares)
            for i in range(int(abs(self.shares))):
                if self.shares >= 0:
                    begin_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
                else:
                    begin_asset += POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])
                
            self.df.loc[self.index, 'balance'] = self.balance
            self.df.loc[self.index, 'asset'] = begin_asset
            self.df.loc[self.index, 'shares'] = self.shares
            
            #buy
            if action > 0:
                #long
                if self.shares >= 0:
                    available_amount = self.balance // (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT)
                    self.balance -= (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(available_amount, action)
                    self.df.loc[self.index+1, 'action'] = int(min(available_amount, action))
                    
                    self.shares += int(min(available_amount, action))
                    for i in range(int(min(available_amount, action))):
                        self.last_bid.append(self.df.iloc[self.index]['close']) 
                    
                #short
                else:
                    self.balance += CONTRACT_PRICE * min(action, abs(self.shares))
                    self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(action, abs(self.shares))
                    self.df.loc[self.index+1, 'action'] = int(min(action, abs(self.shares)))
                    
                    for i in range(int(min(action, abs(self.shares)))):
                        self.balance += (self.last_bid.pop(0) - self.df.iloc[self.index]['close']) * POINT_VALUE
                    
                    self.shares += min(action, abs(self.shares))

            #sell
            elif action < 0:
                #long
                if self.shares > 0:
                    self.balance += CONTRACT_PRICE * min(abs(action), self.shares)
                    self.balance -= (self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(abs(action), self.shares)
                    self.df.loc[self.index+1, 'action'] = -1 * int(min(abs(action), self.shares))
                    
                    for i in range(int(min(abs(action), self.shares))):
                        self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * POINT_VALUE

                    self.shares -= min(abs(action), self.shares)

                else:
                    available_amount = self.balance // (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT)
                    self.balance -= (CONTRACT_PRICE + self.df.iloc[self.index]['close'] * POINT_VALUE * TRANSACTION_FEE_PERCENT) * min(available_amount, abs(action))
                    self.df.loc[self.index+1, 'action'] = -1 * int(min(available_amount, abs(action)))
                    
                    self.shares -= int(min(available_amount, abs(action)))
                    for i in range(int(min(available_amount, abs(action)))):
                        self.last_bid.append(self.df.iloc[self.index]['close'])                     
                    

            self.index += 1
            
            observartion = self.get_obs()
            end_asset = self.balance + CONTRACT_PRICE * abs(self.shares)
            self.df.loc[self.index, 'asset'] = end_asset
            
            for i in range(int(abs(self.shares))):
                if self.shares >= 0:
                    end_asset += POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
                else:
                    end_asset += POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])

            if end_asset <= 0:
                reward = -100
                
                self.df.loc[self.index, 'balance'] = end_asset
                self.df.loc[self.index, 'asset'] = end_asset
                self.balance = end_asset
                self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
                self.df['cumulative return'] = ((1 + (self.df['return'] / 100)).cumprod() - 1) * 100
                
                performance = self.get_performance()
                print(f'date : {self.curr_year}/{self.curr_month}, balance : {self.balance}, total_reward : {self.total_reward}, RoR : {performance[0]}, Sharp : {performance[1]}, MDD : {performance[2]}') 
                done = True
            else: 
                reward = (end_asset - begin_asset) * REWARD_SCALING

            if self.index >= 76:
                self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
                mean_return = self.df['return'][self.index-76:self.index+1].mean() 
                std_return = self.df['return'][self.index-76:self.index+1].std()
                if mean_return != 0 and std_return != 0:
                    Sharp = (mean_return) / std_return
                    reward += Sharp
                    
            self.total_reward += reward

        return observartion, reward, done, False, self.get_info()
    
if __name__ == '__main__':

    pass