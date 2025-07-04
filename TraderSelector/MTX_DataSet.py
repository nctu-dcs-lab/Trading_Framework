import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MTX_Dataset(Dataset):
    def __init__(self, seq_length, start_date, end_date, scaler_norm=None, data_path="MTX_Year_Data_Preprocessed/denoised_data.csv"):
        
        # Filter the data within the specified time range.
        df = pd.read_csv(data_path)
        start_idx = df.index[df['date'] >= start_date][0] if len(df.index[df['date'] >= start_date]) > 0 else None
        end_idx = df.index[df['date'] <= end_date][-1] if len(df.index[df['date'] <= end_date]) > 0 else None
        
        if start_idx is not None and end_idx is not None:
            # 確保 start_idx 之前有至少 100 筆數據
            start_idx = max(0, start_idx - 100)

            # 先從 start_idx 開始切片，然後篩選出符合日期範圍的資料
            filtered_df = df.iloc[start_idx:end_idx + 1]
        else:
            # 若 start_date 或 end_date 不在 df 的範圍內，返回空 DataFrame
            filtered_df = pd.DataFrame(columns=df.columns)            
            
        self.data = torch.tensor(filtered_df['close_denoised'].values, dtype=torch.float32)  # 轉換為 tensor

        # 如果未提供scaler，則訓練集上進行正規化，並保存scaler
        if scaler_norm is None:
            self.normalized_data, self.scaler_norm = self.normalise_data_xlstm(self.data)
        else:
            # 若提供scaler，則使用已有的scaler來轉換數據
            self.normalized_data = scaler_norm.transform(self.data.reshape(-1, 1))
            self.scaler_norm = scaler_norm

        
        self.labels = torch.tensor(filtered_df['trend'].values, dtype=torch.float32).unsqueeze(1)  # 轉換為 tensor
        self.seq_length = seq_length
        
        
        trend_counts = filtered_df.iloc[100:]['trend'].value_counts()
        print(f'shape of data : {self.data.shape}')
        print(f'shape of labels : {self.labels.shape}')
        print(trend_counts)
        
    def __len__(self):
        return len(self.normalized_data) - self.seq_length  # 返回數據集大小

    def __getitem__(self, idx):
        return torch.tensor(self.normalized_data[idx:idx+self.seq_length], dtype=torch.float32), self.labels[idx+self.seq_length-1]  # 返回一組 (數據, 標籤)
    
    def normalise_data_xlstm(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
    
        return scaler.fit_transform(data.reshape(-1, 1)), scaler   
    
    def standardize_data_xlstm(self, data):

        standard_scaler = StandardScaler()
        standardized_data = standard_scaler.fit_transform(data.reshape(-1, 1))
        
        return standardized_data, standard_scaler   

        
if __name__ == "__main__":
    train_dataset = MTX_Dataset(100, '2010-01-04', '2020-6-18')
    val_dataset = MTX_Dataset(100, '2020-6-19', '2020-12-16',scaler_norm=train_dataset.scaler_norm)
    test_dataset = MTX_Dataset(100, '2020-12-17', '2021-6-16', scaler_norm=train_dataset.scaler_norm)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # validation/test 不用 shuffle
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # plot_density(train_dataset.normalized_data, test_dataset.normalized_data, val_dataset.normalized_data)

        
    for batch_x, batch_y in train_loader:
        print(f'shape of batch x : {batch_x.shape}')
        print(f'{batch_x}')
        print(f'shape of batch y : {batch_y.shape}')
        print(f'{batch_y}')
        break