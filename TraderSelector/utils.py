
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
def read_and_format_csv(input_file_path):
    """
    Reads the CSV file and formats the columns.

    Args:
        input_file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The dataframe with formatted columns.
    """
    df = pd.read_csv(input_file_path, engine='python', encoding='cp950',index_col=False)

    # Rename columns
    if "交易日期" in df.columns:
        df.rename(columns={"交易日期" : "date"}, inplace=True)
    if "契約" in df.columns:
        df.rename(columns={"契約" : "stock"}, inplace=True)
    if "到期月份(週別)" in df.columns:
        df.rename(columns={"到期月份(週別)" : "contract month"}, inplace=True)
    if "開盤價" in df.columns:
        df.rename(columns={"開盤價" : "open"}, inplace=True)
    if "最高價" in df.columns:
        df.rename(columns={"最高價" : "high"}, inplace=True)
    if "最低價" in df.columns:
        df.rename(columns={"最低價" : "low"}, inplace=True)
    if "收盤價" in df.columns:
        df.rename(columns={"收盤價" : "close"}, inplace=True)
    if "成交量" in df.columns:
        df.rename(columns={"成交量" : "volume"}, inplace=True)

    # Select relevant columns
    selected_columns = ['date', 'stock', 'contract month', 'open', 'high', 'low', 'close', 'volume']
    df = df[selected_columns]
    df = df[(df['stock'] == 'MTX')]
    df = df[~df['contract month'].str.contains('W|/', na=False)]
    # 將"-"變成NaN
    df.replace('-', pd.NA, inplace=True)
    # 將NaN丟掉
    df = df.dropna()
    
    
    df['date'] = pd.to_datetime(df['date'])
    # 只留下近月
    df['date_ym'] = df['date'].dt.strftime('%Y%m')
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.groupby('date').apply(filter_data).reset_index(drop=True)

    # 合併盤後和一般
    df = df.groupby('date').agg(
        stock=('stock', 'first'),
        contract=('contract month','first'),
        open=('open', 'last'),  
        high=('high', 'max'), 
        low=('low', 'min'), 
        close=('close', 'first'),  
        volume=('volume', 'sum')
    ).reset_index()

    return df


# 按照日期分组，并根据年份条件处理
def filter_data(group):
    if group['date'].dt.year.iloc[0] >= 2018:
        # 如果是2018年及以后，保留前两个相同日期的数据
        return group.head(2)
    else:
        # 如果是2018年以前，保留第一个数据
        return group.head(1)


def convert_column_types(df):
    """
    Converts the column types of the dataframe to appropriate formats.

    Args:
        df (pd.DataFrame): The dataframe to convert.

    Returns:
        pd.DataFrame: The dataframe with converted column types.
    """
  
    df = df.astype({
        'stock': 'str',
        'contract': 'str',
        'date': 'str',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32'
    })

    df = df.reset_index(drop=True)
    # 將str欄位的空格去掉
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    

    return df


def add_columns(input_file_path):
    """
    Reads the CSV, processes it by calling separate functions for each task, 
    and returns the filtered dataframe.

    Args:
        input_file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The filtered and processed dataframe.
    """
    df = read_and_format_csv(input_file_path) 

    df = convert_column_types(df)


    return df


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


def add_EMA(df, ema_len):
    ema = calculate_ema(df['close'], ema_len)
    df.insert(len(df.columns), f'ema{ema_len}', ema)
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

def add_trend(df):
    trend = [0] * len(df['close'])
    for i in range(0, len(df['close'])):
        if df.iloc[i]['open'] <= df.iloc[i]['close']:
            trend[i] = 1
        else:
            trend[i] = 0
            
    df.insert(len(df.columns), 'trend', trend)
    return df 


def plot_density(train_data, test_data, valid_data=None, title="Density Plot of Train, Test, and Valid Data"):
    """
    繪製訓練集、測試集和可選的驗證集的密度圖。
    
    參數:
    - train_data: 訓練集數據（如 numpy array 或 pandas Series）
    - test_data: 測試集數據（如 numpy array 或 pandas Series）
    - valid_data: 驗證集數據（如 numpy array 或 pandas Series），默認為 None
    - title: 圖片標題，默認為 "Density Plot of Train, Test, and Valid Data"
    """
    plt.clf()
    plt.figure(figsize=(10, 6))

    # 訓練集的密度圖
    sns.kdeplot(train_data, shade=True, label='Train Data', color='blue')

    # 測試集的密度圖
    sns.kdeplot(test_data, shade=True, label='Test Data', color='orange')

    # 如果有驗證集，則也畫出
    if valid_data is not None:
        sns.kdeplot(valid_data, shade=True, label='Valid Data', color='green')

    # 設定標題和標籤
    plt.title(title)
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend(loc='upper right')

    # 顯示圖形
    plt.savefig(f'density.png')
    
    
def plot_logits_distribution(logits, logits_type="raw", num_classes=2, bins=50, mode='train'):
    """
    繪製 logits 分布
    
    參數:
        logits (Tensor or ndarray): 模型輸出的 logits
        logits_type (str): 可選 "raw" (原始 logits), "sigmoid" (二分類), "softmax" (多分類)
        num_classes (int): 類別數，適用於多分類 softmax
        bins (int): 直方圖的分箱數
        
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    
    if logits_type == "sigmoid":  # 二分類 Sigmoid 機率分布
        probs = F.sigmoid(torch.tensor(logits)).numpy()
        plt.hist(probs, bins=bins, alpha=0.7, color='g', edgecolor='black')
        plt.xlabel("Probability (Sigmoid Output)")
        plt.title("Distribution of Sigmoid Probabilities")
        
    elif logits_type == "softmax":  # 多分類 Softmax 機率分布
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        pred_probs = probs.max(axis=1)  # 取最大機率的類別
        plt.hist(pred_probs, bins=bins, alpha=0.7, color='r', edgecolor='black')
        plt.xlabel("Max Softmax Probability")
        plt.title("Distribution of Max Predicted Probabilities")
        
    else:  # 原始 logits 分布
        flattened_logits = logits.flatten()
        plt.hist(flattened_logits, bins=bins, alpha=0.7, color='b', edgecolor='black')
        plt.xlabel("Logit Value")
        plt.title("Distribution of Logits")

    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f'{logits_type}_distribution_{mode}.png')
