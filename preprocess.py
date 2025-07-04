from argparse import ArgumentParser
import pandas as pd
import mplfinance as mpf
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
START_YEAR = 2018
END_YEAR = 2024



def find_min_max(df):
    """
    Finds and prints the maximum and minimum values for close, volume, RSI, MACD, CCI, and ADX in the dataframe
    up to the contract month of 202212.

    Args:
        df (pd.DataFrame): The input dataframe with columns including 'contract month', 'close', 'volume', 
                           'rsi', 'macd', 'cci', and 'adx'.
    """
    index = 0
    for i in range(len(df['contract month'])):
        if int(df['contract month'][i]) > 202409:
            break
        else:
            index += 1

    print(f'max close : {np.max(df["close"][:index])}, min close : {np.min(df["close"][:index])}')
    print(f'max volume : {np.max(df["volume"][:index])}, min volume : {np.min(df["volume"][:index])}')
    print(f'max rsi : {np.max(df["rsi"][:index])}, min rsi : {np.min(df["rsi"][:index])}')
    print(f'max macd : {np.max(df["macd"][:index])}, min macd : {np.min(df["macd"][:index])}')
    print(f'max cci : {np.max(df["cci"][:index])}, min cci : {np.min(df["cci"][:index])}')
    print(f'max adx : {np.max(df["adx"][:index])}, min adx : {np.min(df["adx"][:index])}')

def draw_hist(x, name, reverse=False):
    """
    Draws and saves a histogram for the given data.

    Args:
        x (pd.Series or np.ndarray): The data to plot the histogram for.
        name (str): The name to use for the saved histogram image file.
    """
    plt.clf()
    plt.hist(x, bins=30, edgecolor='black', range=(x.min(), x.max()))
    plt.title('Histogram of Random Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    save_dir = 'histogram'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if reverse == True:
        save_path = os.path.join(save_dir,f'{name}-r.png')
    else:    
        save_path = os.path.join(save_dir,f'{name}.png')
    plt.savefig(save_path, format='png')
    plt.close()

def calculate_ma(arr, length):
    """
    Calculates the Moving Average (MA) for the given data and length.

    Args:
        arr (np.ndarray): The input data array.
        length (int): The length for calculating the MA.

    Returns:
        np.ndarray: The calculated MA values.
    """
    ma = []
    for i in range(len(arr)):
        if i < length-1:
            ma.append(arr[i])
        else:
            ma.append(np.mean(arr[i+1-length:i+1]))
            
    return np.array(ma)

def calculate_ema(arr, window_size):
    """
    Calculates the Exponential Moving Average (EMA) for the given data and window size.

    Args:
        arr (np.ndarray): The input data array.
        window_size (int): The window size for calculating the EMA.

    Returns:
        np.ndarray: The calculated EMA values.
    """
    alpha = 2 / (window_size + 1)
    ema = []
    
    for i in range(len(arr)):
        if i == 0:
            ema.append(arr[i])
        else:
            ema.append(ema[i-1] * (1 - alpha) + arr[i] * alpha)
            
    return np.array(ema)


def add_columns(input_file_path):
    """
    Reads the input CSV file, adds and formats necessary columns.

    Args:
        input_file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = pd.read_csv(input_file_path)
        
    first_row = df.columns.tolist()
    df.loc[-1] = first_row
    df.index = df.index + 1
    df = df.sort_index()
    df.columns=['stock', 'contract month', 'date', 'open', 'high', 'low','close','volume']
    df = df.astype({'stock':'str', 'contract month':'str', 'date':'str', 'open':'float32', 'high':'float32', 'low':'float32','close':'float32','volume':'float32'})
    df['date'] = pd.to_datetime(df['date'])
    return df

def add_MA(df, ma_len):
    """
    Adds a moving average (MA) column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'close' column with price data.
        len (int): The window size for calculating the moving average.

    Returns:
        pd.DataFrame: The dataframe with an added column named 'ma{len}', representing the moving average of the 'close' column.

    """
    ma = calculate_ma(df['close'], ma_len)
    df.insert(len(df.columns), f'ma{ma_len}', ma)
    
    return df

def add_EMA(df, ema_len):
    ema = calculate_ema(df['close'], ema_len)
    df.insert(len(df.columns), f'ema{ema_len}', ema)
    return df

def add_trend(df, trend_len, ma):
    """
    Adds the trend column to the dataframe.
    Trend indicates an upward or downward trend.
    
    Args:
        df (pd.DataFrame): The input dataframe with a 'close' and 'ma20' columns.

    Returns:
        pd.DataFrame: The dataframe with the added trend column.
    """   
    # ma method
    # trend = []
    # for i in range(len(df['close'])):
    #     if df.iloc[i]['close'] >= df.iloc[i][f'ma{ma}']:
    #         trend.append(1)
    #     else:
    #         trend.append(-1)
            
    # for i in range(0, len(trend), trend_len):
    #     if sum(trend[i:i+trend_len]) >= 0:
    #         if i+trend_len > len(trend):
    #             trend[i:] = [1] * (len(trend) - i)
    #         else:
    #             trend[i:i+trend_len] = [1] * trend_len
    #         # print(f'index {i} to index {i+trend_len-1} is upward.')
    #     else:
    #         if i+trend_len > len(trend):
    #             trend[i:] = [-1] * (len(trend) - i)
    #         else:
    #             trend[i:i+trend_len] = [-1] * trend_len
    #         # print(f'index {i} to index {i+trend_len-1} is downward.')

    # daily trend method
    trend = [0] * len(df['close'])
    for i in range(0, len(df['close']), trend_len):
        if i+trend_len > len(df['close']):
            if df.iloc[i]['open'] < df.iloc[-1]['close']:
                trend[i:] = [1] * (len(trend) - i)
            else:
                trend[i:] = [-1] * (len(trend) - i)                
        else:
            if df.iloc[i]['open'] < df.iloc[i+trend_len-1]['close']:
                trend[i:i+trend_len] = [1] * trend_len
            else:
                trend[i:i+trend_len] = [-1] * trend_len          
    df.insert(len(df.columns), 'trend', trend) 
    
    return df

def add_MACD(df):
    """
    Adds the MACD column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with a 'close' column.

    Returns:
        pd.DataFrame: The dataframe with the added MACD column.
    """
            
    ema_12 = calculate_ema(df['close'], 12)
    ema_26 = calculate_ema(df['close'], 26)
    macd = ema_12 - ema_26
    
    df.insert(len(df.columns), 'macd', macd)

    return df

def add_RSI(df, n=14):
    """
    Adds the RSI column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with a 'close' column.
        n (int): The period for calculating the RSI.

    Returns:
        pd.DataFrame: The dataframe with the added RSI column.
    """
    up = [0]
    down = [1e-10]
    
    change = 0
    for i in range(1, len(df['close'])):
        change = df['close'][i] - df['close'][i-1]
        if change > 0:
            up.append(change)
            down.append(1e-10)
        elif change < 0:
            up.append(0)
            down.append(abs(change))
        else:
            up.append(0)
            down.append(1e-10)
    
    
    ave_up = calculate_ema(up, n)
    ave_down = calculate_ema(down, n)
    

    rs = ave_up / ave_down
    rsi = 100 - 100 / ( 1 + rs)

    df.insert(len(df.columns), 'rsi', rsi)
    
    return df

def add_CCI(df, n=14):
    """
    Adds the CCI column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with 'high', 'low', and 'close' columns.
        n (int): The period for calculating the CCI.

    Returns:
        pd.DataFrame: The dataframe with the added CCI column.
    """
    tp = []
    md = []
    ma = []
    cci = [0] * 26
    for i in range(len(df['close'])):
        tp.append((df['high'][i] + df['low'][i] + df['close'][i]) / 3)
    
    for i in range(n-1, len(tp)):
        ma.append(np.mean(tp[i-n+1:i+1]))
              
    temp = abs(np.array(tp[n-1:]) - np.array(ma))


    for i in range(n-1, len(temp)):
        md.append(np.mean(temp[i-n+1:i+1]))



    cci = np.append(cci, (np.array(tp[n*2-2:]) - np.array(ma[n-1:])) / (0.015 * np.array(md)))
    df.insert(len(df.columns), 'cci', cci)

    return df

def add_ADX(df,n=14):
    """
    Adds the ADX column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with 'high', 'low', and 'close' columns.
        n (int): The period for calculating the ADX.

    Returns:
        pd.DataFrame: The dataframe with the added ADX column.
    """
    DMP = [1e-10]
    DMM = [1e-10]
    TR = [1e-10]
    for i in range(1, len(df['high'])):
        DMP.append(df['high'][i] - df['high'][i-1] if (df['high'][i] - df['high'][i-1]) > 0 else 0)
        DMM.append(df['low'][i-1] - df['low'][i] if (df['low'][i-1] - df['low'][i]) > 0 else 0)
        TR.append(np.max([df['high'][i] - df['low'][i], 
                          abs(df['high'][i] - df['close'][i-1]),
                          abs(df['low'][i]-df['close'][i-1])]))

    smooth_DMP = calculate_ema(DMP, n)
    smooth_DMM = calculate_ema(DMM, n)
    smooth_TR = calculate_ema(TR, n)
    
    DIP = (smooth_DMP / smooth_TR) * 100
    DIM = (smooth_DMM / smooth_TR) * 100
    
    DX = (abs(DIP - DIM) / (DIP + DIM)) * 100
       
    ADX = calculate_ema(DX, n)
    df.insert(len(df.columns), 'adx', ADX)

    return df

def split_csv(df, scalar, save_dir, trend_len, ma, reverse=False):
    """
    Splits the dataframe into multiple CSV files based on contract month and saves them.

    Args:
        df (pd.DataFrame): The input dataframe with a 'contract month' column.
        scalar (int): The scalar value used in the filename of the saved CSV files.
    """
    
    curr_contract_month = df['contract month'][0]
    start_index = 0
    end_index = 0
    for contract_month in df['contract month']:
        print(f'{str(contract_month)}')
        print(f'{curr_contract_month}')
        if curr_contract_month == str(contract_month):
            end_index += 1
        else:
            split_df = df.iloc[start_index:end_index]
            start_index = end_index
            end_index += 1
            last_contract_month = curr_contract_month
            curr_contract_month = df['contract month'][end_index]
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            if reverse == True:
                save_path = os.path.join(save_dir, f'tfe-mtx00-{last_contract_month}-{scalar}min-r.csv')
            else:
                save_path = os.path.join(save_dir, f'tfe-mtx00-{last_contract_month}-{scalar}min.csv')
            split_df = add_trend(split_df, trend_len, ma)
            split_df.to_csv(save_path, index=False)



    split_df = df.iloc[start_index:end_index]
    start_index = end_index
    end_index += 1
    last_contract_month = curr_contract_month
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    if reverse == True:
        save_path = os.path.join(save_dir, f'tfe-mtx00-{last_contract_month}-{scalar}min-r.csv')
    else:
        save_path = os.path.join(save_dir, f'tfe-mtx00-{last_contract_month}-{scalar}min.csv')
    split_df = add_trend(split_df, trend_len, ma)
    split_df.to_csv(save_path, index=False)
    
    
def reverse_data(df):
    df['open'], df['close'] =  df['close'], df['open']
    df['open'] = df['open'][::-1].values
    df['close'] = df['close'][::-1].values
    df['high'] = df['high'][::-1].values
    df['low'] = df['low'][::-1].values

    return df

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scalar', default=15, type=int, help='The time scalar of market data.')
    parser.add_argument('-r', '--reverse', action='store_true', help='Generate reversal data or not')
    parser.add_argument('--save_dir', default='processed_data', help='The directory to save preprocessed data.')
    parser.add_argument('--trend_len', default=20, type=int, help='Specifies how many candlesticks define a single trend.')
    parser.add_argument('--train_end', default=2020, type=int)
    parser.add_argument('--ma', type=int)
    args = parser.parse_args()
    
    loop_num = 1
    if args.reverse == True:
        loop_num += 1
        
    print(f'start data preprocessing.')
    for loop in range(loop_num):        
        df_list = []
        for year in range (START_YEAR, END_YEAR+1):
            input_file_path = os.path.join('tfe-mtx00', f'tfe-mtx00-{year}-{args.scalar}min.csv')        

            df = add_columns(input_file_path)
            df_list.append(df)
            if loop == 1 and year == args.train_end:
                break
                
        df = pd.concat(df_list, ignore_index=True)
        
        if loop == 1:
            df = reverse_data(df)
        df = add_MA(df, 20)
        df = add_MA(df, 60)
        df = add_MA(df, 120)

        df = add_MACD(df)
        df = add_RSI(df)
        df = add_CCI(df)
        df = add_ADX(df)
        
        find_min_max(df)
        for col in ['close', 'volume', 'macd', 'rsi', 'cci', 'adx']:
            if loop == 1:
                draw_hist(df[col], col, args.reverse)
            else:
                draw_hist(df[col], col)  
        if loop == 1:   
            split_csv(df, args.scalar, trend_len=args.trend_len, ma=args.ma, save_dir=args.save_dir, reverse=args.reverse)
        else:
            split_csv(df, args.scalar, trend_len=args.trend_len, ma=args.ma, save_dir=args.save_dir)
    print(f'Data preprocessing completed.')
        
        
    
    
    


    
