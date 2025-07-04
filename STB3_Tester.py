import gymnasium as gym
import StockEnv
from StockEnv.envs import StockEnvV2
import numpy as np
import torch
import random

import time
from argparse import ArgumentParser
from stable_baselines3.common.policies import obs_as_tensor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
                
def make_env(env_name, save_dir, start_year, end_year, train_start, train_end, init_balance=1e6):
    # environment
    env_id = env_name
    env = gym.make(env_id, save_dir=save_dir, start_year=start_year, end_year=end_year, train_start=train_start, train_end=train_end, init_balance=init_balance)
        
    return env

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, help='The random seed of the system.')
    parser.add_argument('--env', type=str, help='The name of the env.')
    parser.add_argument('--balance', type=int, help='initial balance of agent.')
    parser.add_argument('--trend_len', type=int, default=20, help='Specifies how many candlesticks define a single trend.')
    parser.add_argument('--ma', type=int)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # set env and trader
    env = make_env(args.env, save_dir=f'valid_csv/onlyshort_2021_2024_seed_{args.seed}_DailyMethod', start_year=2021, end_year=2024, train_start=2018, train_end=2020, init_balance=args.balance)
    long_trader = MaskablePPO.load(f'trained_model/experiment_20250306_38.zip')
    short_trader = MaskablePPO.load(f'trained_model/experiment_20250306_39.zip')
    
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
    

    for j in range(48):
        obs, info = env.reset()
        done = False
        bid_price = []
        
        curr_trader = -1
        while not done:
      
            # Choose which trader to use.
            if curr_trader == 1:
                action, state = long_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask())
            elif curr_trader == -1:
                action, state = short_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask())                   
            # Execute action
            obs, reward, done, _, info = env.step(action)
            rewards[j] += reward
            

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

    print(f'ROI : {ROI}')
    print(f'IRR : {IRR}')
    print(f'Ann Vol : {AVOL}')
    print(f'Sharp ratio : {(IRR - 0.017) / AVOL}')
    print(f'cost : {cost}')
    print(f'asset : {asset}')
