import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


months = list(range(1,13))
years = list(range(2019,2020))
file_list = ['onlylong_2021_2024_seed_1_DailyMethod', 'onlyshort_2021_2024_seed_1_DailyMethod']
for year in years:
    for month in months:
        # # plot return
        # # 讀 long 的資料
        # file_path = f'valid_csv/{file_list[0]}/tfe-mtx00-{year}{month:02d}-15min.csv'
        # df = pd.read_csv(file_path)
        # cumulative_return_long = df['cumulative return'].tolist()
                
        # # 讀 short 的資料
        # file_path = f'valid_csv/{file_list[1]}/tfe-mtx00-{year}{month:02d}-15min.csv'
        # df = pd.read_csv(file_path)
        # cumulative_return_short = df['cumulative return'].tolist()
                
        # # x 軸 = K 棒 index
        # x = list(range(len(cumulative_return_long)))

        # # 設定字體與解析度
        # plt.rcParams['font.family'] = 'Arial'
        # plt.figure(figsize=(8, 6))

        # # 畫圖
        # plt.plot(x, cumulative_return_long, label='Long Agent', color='blue')
        # plt.plot(x, cumulative_return_short, label='Short Agent', color='orange')
        
        # # 圖表標題與座標
        # plt.title(f'Cumulative Returns - {year}/{month:02d}', fontsize=14, color='black')
        # plt.xlabel('Index of 15-min Candlesticks (K-bars)', fontsize=12, color='black')
        # plt.ylabel('Cumulative Return', fontsize=12, color='black')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # # 儲存高解析圖
        # plt.savefig(f'cumulativeGraph/DailyMethod_OnlyOnePosition/{year}_{month:02d}_return_500k_new.png', dpi=300)
        # plt.clf()
        
        
        # # plt trend
        file_path = f'synthetic_data_dailyMethod/tfe-mtx00-{year}{month:02d}-15min-r.csv'
        df = pd.read_csv(file_path)

        close = df['close'].tolist()
        ma20 = df['ma20'].tolist()
        ma60 = df['ma60'].tolist()
        ma120 = df['ma120'].tolist()
        trend = df['trend'].tolist()

        x = list(range(len(close)))  # x 軸為 K 棒 index

        last_trend = trend[0]
        start_idx = 0

        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(8, 6))

        for i in range(1, len(close)):
            if trend[i] != last_trend or i == (len(close)-1):
                plt.plot(x[start_idx:i+1], close[start_idx:i+1], color='blue')
                start_idx = i
                last_trend = trend[i]

        plt.title(f'Price Trend Segmentation - {year}/{month:02d}', fontsize=14)
        plt.xlabel('Index of 15-min Candlesticks (K-bars)', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # 儲存圖像，dpi=300 提升解析度
        plt.savefig(f'trendGraph/{year}_{month:02d}_trendlen_60_ma120-r.png', dpi=300)
        plt.clf()


