from .StockEnvV2 import *
from .StockEnvSyn import *
from .StockOnlyOneShareV6 import *


class StockMarketOnlyOneShareV6_2(StockMarketOnlyOneShareV6):

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
                    if i == 1:
                        file_path = f'synthetic_data_dailyMethod/tfe-mtx00-{year}{month:02d}-{self.scalar}min-r.csv'
                    else:
                        file_path = f'synthetic_data_dailyMethod/tfe-mtx00-{year}{month:02d}-{self.scalar}min.csv'
                    temp_df = pd.read_csv(file_path)
                    whole_df = pd.concat([whole_df, temp_df], axis=0, ignore_index=True)
        return whole_df     
    
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
            file_path = f'synthetic_data_dailyMethod/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min.csv'
        elif self.reverse == 1:
            file_path = f'synthetic_data_dailyMethod/tfe-mtx00-{self.curr_year}{self.curr_month:02d}-{self.scalar}min-r.csv'
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