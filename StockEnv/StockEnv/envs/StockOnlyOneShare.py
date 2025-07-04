from .StockEnvV2 import *

class StockMarketOnlyOneShare(StockMarketLongDiscreteMask):
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
            
        return observation, reward, done, False, self.get_info()

    def process_trade(self, action):
        """Executes a trade based on the action."""

        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            # Available number of units traded
            if self.shares == 0:
                available_amount = 1
            else:
                available_amount = 0
                
            actual_action = int(min(available_amount, action))
            self.shares += actual_action
            
            # Record the bid price
            for _ in range(actual_action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.df.iloc[self.index]['close'])
        
        elif action < 0: # sell
            actual_action = int(min(abs(action), self.shares))
            for _ in range(actual_action):
                bid_price = self.last_bid.pop(0)
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
        for i in range(len(self.last_bid)):
            bid_price = self.last_bid.pop(0)
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