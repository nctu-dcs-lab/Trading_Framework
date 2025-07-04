from .StockOnlyOneShareV5 import *

class StockMarketNoLimit2024(StockMarketOnlyOneShareV5):
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
        if self.reverse == 0:
            file_path = f'datasetFor2024/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv'
        elif self.reverse == 1:
            file_path = f'datasetFor2024/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min-r.csv'
        print(f'using {file_path}.')
        print(f'curr year : {self.curr_year}, curr month : {self.curr_month}, contract price : {self.CONTRACT_PRICE}')
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
    
    def load_entire_data(self):
        """
        Loads the entire training dataset for normalization.

        Returns:
            pd.DataFrame: Concatenated DataFrame of the entire dataset for the specified date range.
        """
        whole_df = pd.DataFrame()
        for i in range(2):
            for year in range(self.train_start, self.train_end + 1):
                for month in range(1, 13):
                    if month==9 and year==2024:
                        break
                    # print('ji')
                    if i == 1:
                        file_path = f'datasetFor2024/tfe-mtx00-{year}{month:02d}-{self.scalar}min-r.csv'
                    else:
                        file_path = f'datasetFor2024/tfe-mtx00-{year}{month:02d}-{self.scalar}min.csv'
                    temp_df = pd.read_csv(file_path)
                    # print(temp_df.head())
                    whole_df = pd.concat([whole_df, temp_df], axis=0, ignore_index=True)
        return whole_df       
    
        
    def reset_environment(self):
        """Resets month and year for simulation."""
        if self.curr_month is None or self.curr_year is None:
            self.curr_year = self.start_year
            self.curr_month = 1
            self.reverse = 0
        else:
            self.reverse += 1
            if self.reverse > 1:
                self.reverse = 0
                self.curr_month = (self.curr_month % 12) + 1 
                if self.curr_month == 9 and self.curr_year == 2024:
                    self.curr_month = 1
                if self.curr_month == 1:
                    self.curr_year += 1
                    if self.curr_year > self.end_year:
                        self.curr_year = self.start_year
                        if self.start_year == 2018 or self.start_year == 2019:
                            self.CONTRACT_PRICE = 24000
                        elif self.start_year == 2020:
                            self.CONTRACT_PRICE = 22750
                        elif self.start_year == 2021:
                            self.CONTRACT_PRICE = 33250
                            self.INITIAL_MARGIN = 33250
                            self.MAINTENANCE_MARGIN = 25500
                        elif self.start_year == 2022 or self.start_year == 2023:
                            self.CONTRACT_PRICE = 46000
                            self.INITIAL_MARGIN = 46000
                            self.MAINTENANCE_MARGIN = 35250
                        elif self.start_year == 2024:
                            self.CONTRACT_PRICE = 41750
                            self.INITIAL_MARGIN = 41750
                            self.MAINTENANCE_MARGIN = 32000

        
