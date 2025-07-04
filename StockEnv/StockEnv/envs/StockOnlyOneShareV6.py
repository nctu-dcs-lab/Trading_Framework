from .StockEnvV2 import *
from .StockEnvSyn import *

class StockMarketOnlyOneShareV6(StockMarketSyn):
    def setup_spaces(self):
        self.action_space = spaces.Discrete(201)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5*self.WINDOW_SIZE+2,))
        
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0

        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]
        
        # process trade
        reward = self.process_trade(action)
        self.total_reward += reward
        
        #get next obs
        self.index += 1
        observation = self.get_observation()
        
        
        #record some useful info
        self.record_info(self.balance, action, reward)

        if self.index >= len(self.df) - 1:
            done = True
            reward += self.finalize_episode(self.balance)
            self.total_reward += reward
            print(f'total reward : {self.total_reward}')  
            
        # update Contract Price 
        self.update_contract_price()
        
        return observation, reward, done, False, self.get_info()

    def update_contract_price(self):
        if (self.curr_year == 2018):
            self.CONTRACT_PRICE = 24000
        elif(self.df.iloc[self.index]["date"] == "2019-02-20 13:30:00"):
            self.CONTRACT_PRICE = 26750
        elif(self.df.iloc[self.index]["date"] == "2019-07-08 13:30:00"):
            self.CONTRACT_PRICE = 23500
        elif(self.df.iloc[self.index]["date"] == "2019-10-02 13:30:00"):
            self.CONTRACT_PRICE = 25750
        elif(self.df.iloc[self.index]["date"] == "2019-10-21 13:30:00"):
            self.CONTRACT_PRICE = 22750
        elif(self.df.iloc[self.index]["date"] == "2020-01-30 13:30:00"):
            self.CONTRACT_PRICE = 27500
        elif(self.df.iloc[self.index]["date"] == "2020-03-13 13:30:00"):
            self.CONTRACT_PRICE = 30500
        elif(self.df.iloc[self.index]["date"] == "2020-03-17 13:30:00"):
            self.CONTRACT_PRICE = 33500
        elif(self.df.iloc[self.index]["date"] == "2020-03-20 13:30:00"):
            self.CONTRACT_PRICE = 37000
        elif(self.df.iloc[self.index]["date"] == "2020-11-13 13:30:00"):
            self.CONTRACT_PRICE = 33250
        elif(self.df.iloc[self.index]["date"] == "2021-01-06 13:30:00"):
            self.CONTRACT_PRICE = 37500
        elif(self.df.iloc[self.index]["date"] == "2021-02-05 13:30:00"):
            self.CONTRACT_PRICE = 41750
        elif(self.df.iloc[self.index]["date"] == "2021-05-19 13:30:00"):
            self.CONTRACT_PRICE = 46000
        elif(self.df.iloc[self.index]["date"] == "2022-01-26 13:30:00"):
            self.CONTRACT_PRICE = 50750
        elif(self.df.iloc[self.index]["date"] == "2022-02-08 13:30:00"):
            self.CONTRACT_PRICE = 46000
        elif(self.df.iloc[self.index]["date"] == "2023-01-17 13:30:00"):
            self.CONTRACT_PRICE = 50750
        elif(self.df.iloc[self.index]["date"] == "2023-01-31 13:30:00"):
            self.CONTRACT_PRICE = 46000
        elif(self.df.iloc[self.index]["date"] == "2023-07-28 13:30:00"):
            self.CONTRACT_PRICE = 41750
        elif(self.df.iloc[self.index]["date"] == "2024-02-05 13:30:00"):
            self.CONTRACT_PRICE = 46000
        elif(self.df.iloc[self.index]["date"] == "2024-02-16 13:30:00"):
            self.CONTRACT_PRICE = 41750
        elif(self.df.iloc[self.index]["date"] == "2024-03-07 13:30:00"):
            self.CONTRACT_PRICE = 44750
        elif(self.df.iloc[self.index]["date"] == "2024-05-03 13:30:00"):
            self.CONTRACT_PRICE = 49500
        elif(self.df.iloc[self.index]["date"] == "2024-05-17 13:30:00"):
            self.CONTRACT_PRICE = 54500
        elif(self.df.iloc[self.index]["date"] == "2024-06-25 13:30:00"):
            self.CONTRACT_PRICE = 60250
        elif(self.df.iloc[self.index]["date"] == "2024-08-09 13:30:00"):
            self.CONTRACT_PRICE = 66250
        elif(self.df.iloc[self.index]["date"] == "2024-08-22 13:30:00"):
            self.CONTRACT_PRICE = 73000
        elif(self.df.iloc[self.index]["date"] == "2024-09-27 13:30:00"):
            self.CONTRACT_PRICE = 80500
        elif(self.df.iloc[self.index]["date"] == "2024-11-13 13:30:00"):
            self.CONTRACT_PRICE = 80500
                        
    def process_trade(self, action):
        """Executes a trade based on the action."""

        reward = 0
        # Determine whether to buy, sell, or hold
        if action < 0: # sell    
            self.shares -= abs(action)
            # Record the bid price
            for _ in range(abs(action)):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
        
        elif action > 0: # buy
            for _ in range(abs(action)):
                bid_price = self.last_bid.pop(0)
                reward += bid_price - self.df.iloc[self.index]['close']
                if bid_price > self.df.iloc[self.index]['close']:
                    self.win += 1
            self.shares += abs(action)
            
        else: # hold
            pass
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
        return reward
        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * abs(self.shares)
        for i in range(int(abs(self.shares))):
            if self.shares >= 0:
                asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
            else:
                asset +=  self.POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])
        return asset



    def valid_action_mask(self):
        actions_mask = np.ones(201, dtype=int)
        if(self.shares == 0): # only 100, 99 valid
            actions_mask[0:99] = 0
            actions_mask[101:201] = 0
        elif(self.shares < 0): # only 100 and 101 valid
            actions_mask[0:100] = 0
            actions_mask[102:201] = 0
        elif(self.shares > 0): # only 99 and 100 valid
            actions_mask[0:99] = 0
            actions_mask[101:201] = 0 
        
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
        for i in range(len(self.last_bid)):
            bid_price = self.last_bid.pop(0)
            reward +=  bid_price - self.df.iloc[self.index]['close']
            if bid_price > self.df.iloc[self.index]['close']:
                self.win += 1
        
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