from .StockEnvV2 import *

class StockMarketOnlyOneShareTestV2(StockMarketLongDiscreteMask):
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
        observation[0] = self.init_balance / (self.init_balance * 3)
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        
        #record some useful info
        self.record_info(begin_asset, action, reward)

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
                    if self.start_year == 2018 or self.start_year == 2019:
                        self.CONTRACT_PRICE = 24000
                    elif self.start_year == 2020:
                        self.CONTRACT_PRICE = 22750
                    elif self.start_year == 2021:
                        self.CONTRACT_PRICE = 33250
                    elif self.start_year == 2022 or self.start_year == 2023:
                        self.CONTRACT_PRICE = 46000
                    elif self.start_year == 2024:
                        self.CONTRACT_PRICE = 41750
                    

    def process_trade(self, action):
        """Executes a trade based on the action."""
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            # Available number of units traded
            if self.shares == 0:
                available_amount = 1
            else:
                available_amount = 0
                
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
        if(self.shares == 0):
            available_amount = 1
        else:
            available_amount = 0
            
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
            if self.df.iloc[self.index]['close'] > bid_price:
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