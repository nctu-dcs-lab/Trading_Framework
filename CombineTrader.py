import gymnasium as gym
import StockEnv
from StockEnv.envs import StockEnvV2
import numpy as np
import torch
import random
from argparse import ArgumentParser
from sb3_contrib import MaskablePPO
import pandas as pd
from TraderSelector.xLSTM_TS import *
from TraderSelector.MTX_DataSet import *
import yaml
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
                
def make_env(env_name, save_dir, start_year, end_year, train_start, train_end, init_balance=1e6):
    # environment
    env_id = env_name
    env = gym.make(env_id, save_dir=save_dir, start_year=start_year, end_year=end_year, train_start=train_start, train_end=train_end, init_balance=init_balance)
        
    return env

def get_selector_input(date_time, df):
        date = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        date_idx = df[df['date'] == date].index[0]
        selector_input_df = df.iloc[max(0, date_idx - 99): date_idx + 1]
        selector_input = torch.tensor(selector_input_df['close_denoised'].values, dtype=torch.float32)    
        selector_input = selector_input.unsqueeze(0).unsqueeze(-1)
        selector_input = selector_input.to('cuda')
        return selector_input
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, help='The random seed of the system.')
    parser.add_argument('--env', type=str, help='The name of the env.')
    parser.add_argument('--balance', type=int, help='initial balance of agent.')
    parser.add_argument('--model', type=str, help='The model used by the selector')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # set env and trader
    env = make_env(args.env, save_dir=f'valid_csv/CombineTrader_model_{args.model}_seed_{args.seed}_final', start_year=2021, end_year=2024, train_start=2018, train_end=2020, init_balance=args.balance)
    long_trader = MaskablePPO.load(f'trained_model/experiment_20250306_38.zip')
    short_trader = MaskablePPO.load(f'trained_model/experiment_20250306_39.zip')
    
    # initial metric list
    rewards = [0] * 48
    RoR = [0] * 48
    Sharp = [0] * 48
    MDD = [0] * 48
    
    '''
    '''
    cost = args.balance
    profit = 0
    asset = args.balance
    '''
    '''
    
    total_choice_wrong = 0
    total_trend_num = 0
    
    # load trader selector
    # trader_selector = load('TraderSelector/random_forest_model_212.joblib')
    xlstm_stack, input_projection, output_projection = create_xlstm_model(100)
    trader_selector = xLSTMClassifier(input_projection, xlstm_stack, output_projection)
    year_list = ['2021', '2022', '2023', '2024']
    half_year = 0
    year_idx = -1
    
    
    for j in range(48):
        obs, info = env.reset()   
        done = False
        
        # initial parameters
        time_step = env.index
        trend_num = 0
        choice_wrong = 0
        curr_trader = 0


        if j % 6 == 0:        
            trader_selector_df = pd.read_csv("TraderSelector/MTX_Year_Data_Preprocessed/denoised_data.csv") 
            if half_year == 0 or half_year == 2:
                year_idx += 1
                half_year = 1
            elif half_year == 1:
                half_year = 2
            
            config = load_config(f'TraderSelector/config/{year_list[year_idx]}H{half_year}.yaml')
            train_cfg = config["dataset"]["train"]   
            # Load the data used by the selector
            train_dataset = MTX_Dataset(100, train_cfg["start_date"],  train_cfg["end_date"], data_path="TraderSelector/MTX_Year_Data_Preprocessed/denoised_data.csv")                                     
            trader_selector.load_state_dict(torch.load(f'TraderSelector/model/xlstm_classifier_{year_list[year_idx]}H{half_year}.pth')) 
            trader_selector_df['close_denoised'] = train_dataset.scaler_norm.transform(trader_selector_df[['close_denoised']])
       
        trader_selector.to('cuda')
        trader_selector.eval()


        while not done:
            action = 100 # default action is hodl(100).
            # print(f'env index : {env.index}')
            # get curr date time
            date_time = datetime.strptime(env.df.iloc[time_step]["date"], "%Y-%m-%d %H:%M:%S").strftime("%H:%M")
            
            # Set the trend at intervals based on trend_len.
            if (date_time == '13:45'):
                # curr_trader = trader_selector.predict(df[env.index-225:env.index+1].reshape(1,-1))
                trader_selector_input = get_selector_input(env.df.iloc[time_step]["date"], trader_selector_df)
                with torch.no_grad():    
                    pred = trader_selector(trader_selector_input)
                    predicted = (torch.sigmoid(pred) > 0.5).float()
                    predicted = predicted.cpu()
                    if predicted == 1:
                        curr_trader = 1
                    else:
                        curr_trader = -1
                trend_num += 1
                true_trader = env.df.iloc[env.index+1]["trend"]
                # print(f'true trader : {true_trader}')
                # print(f'curr_trader : {curr_trader}')
                if true_trader != curr_trader:
                    choice_wrong += 1
                env.set_trader(curr_trader)

            
            # Choose which trader to use.
            if curr_trader == 1:
                action, state = long_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask_long())
            elif curr_trader == -1:
                action, state = short_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask_short())         
            
            # Execute action
            obs, reward, done, _, info = env.step(action)
            rewards[j] += reward
            time_step += 1
            
        total_trend_num += trend_num
        total_choice_wrong += choice_wrong    
        
        print(f'choice wrong : {choice_wrong}')
        print(f'trend num : {trend_num}')
        print(f'prob of wrong choice : {choice_wrong / trend_num}')
        RoR[j], Sharp[j], MDD[j] = env.get_performance()
        print(f'ROR : {RoR[j]}')
        
        '''
        '''
        profit = info['balance'] - args.balance         
        print(f'asset : {asset}, profit : {profit}')
        print(f'date : {info["date"]}')

        asset += profit
        if asset < args.balance:
            cost += args.balance - asset
            asset = args.balance
        '''
        '''    
        print(f'j = {j}')

    rewards = np.array(rewards)
    RoR = np.array(RoR)
    Sharp = np.array(Sharp)
    MDD = np.array(MDD)


    ROI = (asset-cost)/cost
    IRR = (1+ROI) ** (1/4) - 1
    AVOL = np.std(RoR) * (12 ** 0.5)
    print(f'mean reward : {np.mean(rewards):4.4f}, unbias std_reward : {np.std(rewards, ddof=1):4.4f}, bias std_reward : {np.std(rewards, ddof=0):4.4f}')
    print(f'mean RoR : {np.mean(RoR):4.4f}, std RoR : {np.std(RoR):4.4f}, MDD : {np.min(MDD):4.4f}')
    print(f'total choice wrong : {total_choice_wrong}')
    print(f'total trend num : {total_trend_num}')
    print(f'prob of total wrong choice : {total_choice_wrong / total_trend_num}')
    print(f'ROI : {ROI}')
    print(f'IRR : {IRR}')
    print(f'Ann Vol : {AVOL}')
    print(f'Sharp ratio : {(IRR - 0.017) / AVOL}')
    print(f'cost : {cost}')
    print(f'asset : {asset}')
