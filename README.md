# 環境設置
使用 env.yml 文件來配置 Conda 虛擬環境，以確保開發環境的一致性和可重現性。請按照以下步驟設置環境：
## 步驟 1：安裝 Conda
請確保已安裝 Conda。如未安裝，可從 Anaconda 官網 或 Miniconda 官網 下載並安裝。
## 步驟 2：創建環境
使用提供的 env.yml 文件創建 Conda 環境，該文件定義了項目所需的所有依賴項。運行以下命令創建環境：
conda env create -f env.yml

## 步驟 3：啟動環境
創建完成後，使用以下命令啟動環境：
```
conda activate py39stock
```

## 步驟 4：驗證環境
檢查環境是否正確設置，運行以下命令以查看已安裝的包：
```
conda list
```

# 資料集處理
需要先執行處理資料集才能進行後續的訓練和測試
```
python preprocess.py -r --save_dir synthetic_data_dailyMethod --ma 20 --train_end 2020
```

# 模型訓練
本項目使用 Sliding_window_train.sh 腳本來訓練兩個方向的 agent（long 和 short），採用滑動窗口方法進行模型訓練。訓練範圍由腳本中的 train_start_year 參數決定。訓練過程的日誌、模型檔案及操作細節將分別儲存於以下位置：

訓練過程日誌：train_log
訓練好的模型：trained_model
訓練操作細節：trained_csv

## 步驟 1：準備環境
請確保已按照「環境設置」部分的說明，使用 env.yml 成功創建並啟動 Conda 環境。
## 步驟 2：修改訓練範圍
訓練範圍由 Sliding_window_train.sh 中的 train_start_year 參數控制。預設情況下，訓練範圍涵蓋三年。例如，設置 train_start_year=2018 將使訓練數據範圍為 2018 年至 2020 年。
## 步驟 3：執行訓練腳本
在啟動的 Conda 環境中，運行以下命令以執行訓練：
```
bash Sliding_window_train.sh
```

該腳本將同時訓練 long 和 short 方向的 agent，並根據指定的 train_start_year 自動選取對應的三年數據範圍進行訓練。

## 步驟 4：檢查訓練輸出
訓練完成後，檢查以下輸出：

train_log：儲存訓練過程的詳細日誌，包括損失值、訓練進度等資訊。
trained_model：儲存訓練好的模型檔案，可用於後續推理或評估。
trained_csv：記錄訓練操作細節，例如買賣時機等。

檢查這些文件是否正確生成，並確認其路徑是否與 Sliding_window_train.sh 中的配置一致。

## 注意事項

確保訓練數據文件位於腳本預期的路徑中。
訓練過程可能因硬件配置而耗時較長，請檢查 train_log 以監控進度。
若需調整其他超參數（如窗口大小、學習率等），請參考 Sliding_window_train.sh。
確認 train_log、trained_model 和 trained_csv 的儲存路徑是否已建立。

# 模型測試
本項目使用 STB3_Tester.py 腳本來測試單一模型的效能，針對 long 或 short 方向的 agent 進行評估。測試過程需要指定訓練好的模型檔案路徑、選擇測試方向，以及設置儲存測試結果的目錄。以下為測試步驟：
## 步驟 1：準備環境
請確保已按照「環境設置」部分的說明，使用 env.yml 成功創建並啟動 Conda 環境。
## 步驟 2：修改模型路徑
在 STB3_Tester.py 中，修改以下程式碼以指定要測試的 long 和 short 方向模型的路徑。例如，預設使用以下模型：
```
long_trader = MaskablePPO.load(f'trained_model/experiment_20250306_38.zip')
short_trader = MaskablePPO.load(f'trained_model/experiment_20250306_39.zip')
```

根據需要更新模型檔案路徑。例如，若使用新的模型文件，修改為：
```
long_trader = MaskablePPO.load(f'trained_model/new_model_long.zip')
short_trader = MaskablePPO.load(f'trained_model/new_model_short.zip')
```

確保指定的模型檔案存在於 trained_model 目錄中。
## 步驟 3：設置測試方向
在 STB3_Tester.py 第62行，設置 curr_trader 參數以選擇測試 long 或 short 方向的 agent：
```
curr_trader = 1：測試 long 方向的模型。
curr_trader = -1：測試 short 方向的模型。
```

例如，測試 short 方向的模型，設置為：
```
curr_trader = -1
```
## 步驟 4：修改儲存目錄
在 STB3_Tester.py 中，修改環境設置中的 save_dir 參數以指定測試結果的儲存路徑。例如，預設儲存路徑為：
```
env = make_env(args.env, save_dir=f'valid_csv/onlyshort_2021_2024_seed_{args.seed}_DailyMethod', start_year=2021, end_year=2024, train_start=2018, train_end=2020, init_balance=args.balance)
```
確保指定的 save_dir 路徑已存在。

## 步驟 5：執行測試
在啟動的 Conda 環境中，運行以下命令以執行測試：
```
python STB3_Tester.py
```

該腳本將根據指定的 curr_trader 測試對應方向的模型，並將測試結果儲存至指定的 save_dir。
注意事項

確保 trained_model 目錄中的模型檔案路徑正確，且檔案存在。
確認 save_dir 的儲存路徑已建立，或腳本能自動創建。
若需調整其他參數（如 start_year、end_year 或 init_balance），請參考 STB3_Tester.py 中的 args 設置。

# 框架模擬測試
使用 CombineTest.sh 來模擬不同準確度下框架的效能。

## 步驟 1：修改模型路徑
在 STB3_CombineTest.py 中，修改以下程式碼以指定要測試的 long 和 short 方向模型的路徑。例如，預設使用以下模型：
```
long_trader = MaskablePPO.load(f'trained_model/experiment_20250306_38.zip')
short_trader = MaskablePPO.load(f'trained_model/experiment_20250306_39.zip')
```

根據需要更新模型檔案路徑。例如，若使用新的模型文件，修改為：
```
long_trader = MaskablePPO.load(f'trained_model/new_model_long.zip')
short_trader = MaskablePPO.load(f'trained_model/new_model_short.zip')
```
## 步驟 2：執行測試
```
bash CombineTest.sh
```


# 交易選擇器訓練

## 步驟 1：資料集處裡
```
python DataPreprocessing.py
```
## 步驟 2：執行訓練
```
baxh train_model.sh
```
## 注意事項
修改config裡的yaml檔來設定訓練、驗證、測試範圍。
確保model目錄已經建立

# 框架測試
使用 CombineTrader.py 來測試框架的效能。

## 步驟 1：修改模型路徑
在 CombineTrader.py 中，修改以下程式碼以指定要測試的 long 和 short 方向模型的路徑。例如，預設使用以下模型：
```
long_trader = MaskablePPO.load(f'trained_model/experiment_20250306_38.zip')
short_trader = MaskablePPO.load(f'trained_model/experiment_20250306_39.zip')
```

根據需要更新模型檔案路徑。例如，若使用新的模型文件，修改為：
```
long_trader = MaskablePPO.load(f'trained_model/new_model_long.zip')
short_trader = MaskablePPO.load(f'trained_model/new_model_short.zip')
```

## 步驟 2：執行測試
```
python CombineTrader.py --seed 1 --env StockEnvCombineTrader --balance 500000 --model xlstm
```