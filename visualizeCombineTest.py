import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser


def draw(mean, std, x, ylabel, trend, ma):
    plt.plot(x, mean, marker='o', label='Mean')    
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color='gray',
        alpha=0.3,
        label='STD'
    )
    for i, (xi, yi) in enumerate(zip(x, mean)):
        plt.text(xi, yi, f'{yi:.2f}', ha='center', va='bottom', fontsize=9)
        
    plt.title(f"Trader Selector Simulation - {ylabel}")
    plt.xlabel("Prob")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    folder_path = f'CombinerTestGraph'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/trend_{trend:03d}_ma_{ma:03d}_{ylabel}_100K.png')
    plt.clf()

trend_list = [76]
prob_list = [50, 60, 70, 80, 90, 100]
ma_list = [20]
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# Probability Changes
for trend in trend_list:
    for ma in ma_list:
        ROI_data = []
        IRR_data = []
        SP_data = []
        for seed in seed_list:
            ROI = []
            IRR = []
            SP = []
            for prob in prob_list:
                print(f'trend : {trend}, seed : {seed}, prob : {prob}')                                      
                file_path = f'CombinerTest/seed{seed}/CombineTrader_trend_{trend}_prob_{prob}_ma_{ma}_seed_{seed}_Test_DailyMethod_100K.txt'
                
                with open(file_path, "r") as file:
                    for line in file:
                        line = line.strip()
                        ROI_idx = line.find('ROI')
                        IRR_idx = line.find('IRR')
                        SP_idx = line.find('Sharp')
                        if ROI_idx != -1:
                            ROI.append(float(line[ROI_idx+6:]))
                        if IRR_idx != -1:
                            IRR.append(float(line[IRR_idx+6:]))
                        if SP_idx != -1:
                            SP.append(float(line[IRR_idx+14:]))
            ROI_data.append(ROI)
            IRR_data.append(IRR)
            SP_data.append(SP)
        print(ROI_data)
        print(IRR_data)
        print(SP_data)
        for rr in ROI_data:
            print(len(rr))
        mean_ROI = np.mean(ROI_data, axis=0)
        mean_IRR = np.mean(IRR_data, axis=0)
        mean_SP = np.mean(SP_data, axis=0)
        std_ROI = np.std(ROI_data, axis=0)               
        std_IRR = np.std(IRR_data, axis=0)               
        std_SP = np.std(SP_data, axis=0)    
        print(f'trend {trend}, ma {ma}')
        print(f'mean roi : {mean_ROI}')
        print(f'std roi : {std_ROI}')
        print(f'mean irr : {mean_IRR}')
        print(f'std irr : {std_IRR}')
        print(f'mean sp : {mean_SP}')
        print(f'std sp : {std_SP}')
        print(f'-'*30)
        draw(mean_ROI, std_ROI, prob_list, 'ROI', trend, ma)
        draw(mean_IRR, std_IRR, prob_list, 'IRR', trend, ma)
        draw(mean_SP, std_SP, prob_list, 'SP', trend, ma)
        