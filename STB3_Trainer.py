
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

SEED = 42

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        print(progress)
        return progress * initial_value

    return func

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
    parser.add_argument('--lr', type=float, help='The model\'s learning rate.', default=3e-4)
    parser.add_argument('--bs', type=int, help='The training batch size.', default=8)
    parser.add_argument('--total_timesteps', '-t', type=int, help='The total number of training timesteps.', default=1000000)
    parser.add_argument('--update_freq', type=int, help='Number of model updates per episode.', default=10)
    parser.add_argument('--gamma', type=float, help='Discount factor.', default=0.99)
    parser.add_argument('--steps', type=int, help='The number of steps to run for each environment per update.', default=4096)
    parser.add_argument('--save_name', type=str, help='The name of file to save.')
    parser.add_argument('--env', type=str, help='The name of the env.')
    parser.add_argument('--start_year', type=int, default=2018)
    parser.add_argument('--end_year', type=int, default=2020)
    parser.add_argument('--train_start', type=int, default=2018)
    parser.add_argument('--train_end', type=int, default=2020)
    parser.add_argument('--test_start', type=int, default=2021)
    parser.add_argument('--test_end', type=int, default=2023)
    parser.add_argument('--balance', type=int, help='initial balance of agent.')
    parser.add_argument('-m', '--mask', action='store_true', help='using maskable env.')
    parser.add_argument('--retrain', action='store_true', help='retrain the model.')
    parser.add_argument('--trained_model', type=str, help='name of the retrain model.')
    args = parser.parse_args()
    
    print(f'Training config : ')
    print(f'Learing rate : {args.lr}')
    print(f'Batch size : {args.bs}')
    print(f'Total timesteps : {args.total_timesteps}')
    print(f'Update freq : {args.update_freq}')
    print(f'Gamma : {args.gamma}')
    print(f'Steps : {args.steps}')
    print(f'Initial balance : {args.balance}')
    print(f'Save name : {args.save_name}')
    print(f'Env : {args.env}')
    print(f'='*165)
    
    # train the model    
    seed_everything(SEED)

    
    env = make_env(args.env, save_dir=f'trained_csv/{args.save_name}', start_year=args.start_year, end_year=args.end_year, train_start=args.train_start, train_end=args.train_end, init_balance=args.balance, mask=args.mask) # initializing environment
    start_time = time.time()
    
    if args.retrain:
        model = MaskablePPO.load(f'trained_model/{args.trained_model}.zip', env=env)
        model.ent_coef = 0.005
        model.batch_size = args.bs
        model.learning_rate = linear_schedule(args.lr)
        model.n_epochs = args.update_freq
        model.gamma = args.gamma
        model.n_steps = args.steps
        model.verbose = 1
        model.tensorboard_log = f'train_log/{args.save_name}'
    else:
        model = MaskablePPO('MlpPolicy', 
                    env, 
                    ent_coef=0.005, 
                    batch_size=args.bs, 
                    learning_rate=linear_schedule(args.lr), 
                    n_epochs=args.update_freq, 
                    gamma=args.gamma, 
                    n_steps=args.steps, 
                    verbose=1, 
                    tensorboard_log=f'train_log/{args.save_name}')     
   
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f'trained_model/{args.save_name}')
    end_time = time.time()
    print(f'training time : {end_time-start_time}s')