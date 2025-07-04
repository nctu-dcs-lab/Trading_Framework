import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os

months = list(range(1,13))
years = list(range(2018,2024))

parser = ArgumentParser()
parser.add_argument('--trend_len', default=20, type=int, help='Specifies how many candlesticks define a single trend.')
parser.add_argument('--ma', type=int)
parser.add_argument('--data_path', type=str, default='synthetic_data')
args = parser.parse_args()

for year in years:
    for month in months:

        
        # plt trend
        file_path = f'{args.data_path}/tfe-mtx00-{year}{month:02d}-15min.csv'
        df = pd.read_csv(file_path)
        close = df['close'].tolist()
        ma20 = df['ma20'].tolist()
        ma60 = df['ma60'].tolist()
        ma120 = df['ma120'].tolist()
        trend = df['trend'].tolist()
        x = list(range(0,len(close)))
        
        last_trend = trend[0]
        start_idx = 0
        ma20_label_shown = False
        ma60_label_shown = False
        ma120_label_shown = False
        for i in range(1, len(close)):
            if trend[i] != last_trend or i == (len(close)-1):
                if last_trend == 1:
                    plt.plot(x[start_idx:i+1], close[start_idx:i+1], color='red')
                else:
                    plt.plot(x[start_idx:i+1], close[start_idx:i+1], color='green')
                plt.plot(x[start_idx:i+1], ma20[start_idx:i+1], color='black', label='ma20' if not ma60_label_shown else "")
                plt.plot(x[start_idx:i+1], ma60[start_idx:i+1], color='blue',  label='ma60' if not ma60_label_shown else "")
                plt.plot(x[start_idx:i+1], ma120[start_idx:i+1], color='brown', label='ma120' if not ma60_label_shown else "")
                ma20_label_shown = True
                ma60_label_shown = True
                ma120_label_shown = True            
                
                start_idx = i
                last_trend = trend[i]
        
        plt.legend()
        folder_path = f'trendGraph/CombineTrader_trend_{args.trend_len}_ma_{args.ma}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f'{folder_path}/{year}_{month:02d}.png')

        plt.clf()


