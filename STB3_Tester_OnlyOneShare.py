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

STOP_LOSS = -25
INITIAL_MARGIN = 46000
MAINTENANCE_MARGIN = 35,250
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
                
def make_env(env_name, save_dir, start_year, end_year, train_start, train_end, init_balance=1e6, mask=False):
    # environment
    env_id = env_name
    env = gym.make(env_id, save_dir=save_dir, start_year=start_year, end_year=end_year, train_start=train_start, train_end=train_end, init_balance=init_balance)

    if mask:
        env = ActionMasker(env, mask_fn)
        
    return env

def mask_fn(env):
    return env.valid_action_mask()
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, help='The name of file to save.')
    parser.add_argument('--seed', type=int, help='The random seed of the system.')
    parser.add_argument('--env', type=str, help='The name of the env.')
    parser.add_argument('--balance', type=int, help='initial balance of agent.')
    parser.add_argument('-m', '--mask', action='store_true', help='using maskable env.')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    #Validate the model
    env = make_env(args.env, save_dir=f'valid_csv/{args.model_name}', start_year=2021, end_year=2023, train_start=2018, train_end=2020, init_balance=args.balance, mask=args.mask)
    model = MaskablePPO.load(f'trained_model/{args.model_name}.zip', env=env)
    rewards = [0] * 36
    RoR = [0] * 36
    Sharp = [0] * 36
    MDD = [0] * 36
    
    '''
    '''
    cost = args.balance
    profit = 0
    asset = args.balance
    '''
    '''
    
    for i in range(36):
        obs, info = env.reset()
        done = False
        bid_Price = 0
        while not done:
            if args.mask:
                action, state = model.predict(obs, deterministic=False, action_masks=env.valid_action_mask())
                
            else:
                action, state = model.predict(obs, deterministic=False)

            # log bid price
            if action == 101:
                bid_Price = info['close']
                # input()
            elif action == 99:
                bid_Price = 0
            elif action == 100:
                if (info['close'] - bid_Price) <= STOP_LOSS:
                    action = 99
                    bid_Price = 0
                elif (info['close'] > bid_Price) and bid_Price != 0:
                    bid_Price = info['close']
                    
            state = obs_as_tensor(obs, model.policy.device)
            obs, reward, done, _, info = env.step(action)
            rewards[i] += reward
        RoR[i], Sharp[i], MDD[i] = env.get_performance()
        
        '''
        '''
        profit = info['balance'] - args.balance         
        print(f'asset : {asset}, profit : {profit}')
        asset += profit
        if asset < args.balance:
            cost += args.balance - asset
            asset = args.balance
        '''
        '''    
            
    rewards = np.array(rewards)
    RoR = np.array(RoR)
    Sharp = np.array(Sharp)
    MDD = np.array(MDD)


    ROI = (asset-cost)/cost
    IRR = (1+ROI) ** (1/3) - 1
    AVOL = np.std(RoR) * (12 ** 0.5)
    print(f'mean reward : {np.mean(rewards):4.4f}, unbias std_reward : {np.std(rewards, ddof=1):4.4f}, bias std_reward : {np.std(rewards, ddof=0):4.4f}')
    print(f'mean RoR : {np.mean(RoR):4.4f}, std RoR : {np.std(RoR):4.4f}, MDD : {np.min(MDD):4.4f}')
    print(f'ROI : {ROI}')
    print(f'IRR : {IRR}')
    print(f'Ann Vol : {AVOL}')
    print(f'Sharp ratio : {(IRR - 0.017) / AVOL}')
    print(f'cost : {cost}')
    print(f'asset : {asset}')
