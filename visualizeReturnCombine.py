import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
months = list(range(1,13))
years = list(range(2021,2024))

parser = ArgumentParser()

parser.add_argument('--filename', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()


for year in years:
    for month in months:
        # # plot return
        file_path = f'valid_csv/{args.filename}/tfe-mtx00-{year}{month:02d}-15min.csv'
        df = pd.read_csv(file_path)
        cumulative_return = df['cumulative return'].tolist()
        trader = df['trader'].tolist()
        last_trader = 0
        start_idx = 0
        x = list(range(0,len(cumulative_return)))
        
        for i in range(len(cumulative_return)):
            if trader[i] != 0:
                if last_trader != trader[i] or i == (len(cumulative_return)-1):
                    if last_trader == 0:
                        last_trader = trader[i]
                    if last_trader == 1:
                        plt.plot(x[start_idx:i+1], cumulative_return[start_idx:i+1], color='red') 
                    else:
                        plt.plot(x[start_idx:i+1], cumulative_return[start_idx:i+1], color='green') 
        
                    start_idx = i
                    last_trader = trader[i]
        
        folder_path = f'cumulativeGraph/seed{args.seed}/{args.filename}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f'{folder_path}/{year}_{month:02d}.png')
        plt.clf()
        



