import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader
from MTX_DataSet import *
from xLSTM_TS import *
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from utils import *
import yaml
from argparse import ArgumentParser

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def set_random_seed(seed):
    # 設定 Python 原生的隨機種子
    random.seed(seed)
    
    # 設定 NumPy 的隨機種子
    np.random.seed(seed)
    
    # 設定 PyTorch 的隨機種子
    torch.manual_seed(seed)
    
    # 如果有使用 CUDA，設定 CUDA 隨機數種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多個 GPU，設置所有 GPU 的種子
    
    # 設定 cudnn 的隨機性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def create_dataloader(seq_length, config_path):
    
    config = load_config(config_path)
    batch_size = config["dataloader"]["batch_size"]
    train_cfg = config["dataset"]["train"]
    val_cfg = config["dataset"]["val"]
    test_cfg = config["dataset"]["test"]  
    
      
    train_dataset = MTX_Dataset(seq_length, train_cfg["start_date"], train_cfg["end_date"])
    val_dataset = MTX_Dataset(seq_length, val_cfg["start_date"], val_cfg["end_date"], scaler_norm=train_dataset.scaler_norm)
    test_dataset = MTX_Dataset(seq_length, test_cfg["start_date"], test_cfg["end_date"], scaler_norm=train_dataset.scaler_norm)
   
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # validation/test 不用 shuffle
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'train start date : {train_cfg["start_date"]}, train end date : {train_cfg["end_date"]}')
    print(f'val start date : {val_cfg["start_date"]}, val end date : {val_cfg["end_date"]}')
    print(f'test start date : {test_cfg["start_date"]}, test end date : {test_cfg["end_date"]}')
    print(len(train_loader.dataset))
    print(len(val_loader.dataset)) 
    print(len(test_loader.dataset))    
    return train_loader, val_loader, test_loader


def train_model(xLSTM_classifier, train_loader, val_loader, name):
    # hyperparameters
    lr = 0.001
    num_epochs = 200
    
    best_val_loss = float('inf')
    patience = 40
    trigger_times = 0
    
    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Move the model to the device (GPU or CPU)
    xLSTM_classifier.to(device)    
    
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(xLSTM_classifier.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10) 
    
    # Trainin model
    for epoch in range(num_epochs):
        xLSTM_classifier.train()
        
        correct_train = 0
        total_train = 0      
        all_train_preds = []
        all_train_labels = []
        # logits_list = []          
        for batch_x, batch_y in train_loader:
            # Move data to the selected device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = xLSTM_classifier(batch_x) 
            loss = criterion(pred, batch_y)

            # # log logits
            # logits_list.append(pred.cpu())

            # Calculate accuracy for training
            predicted = (torch.sigmoid(pred) > 0.5).float()  # Threshold at 0.5 for binary classification
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

            optimiser.zero_grad()
            loss.backward()            
            torch.nn.utils.clip_grad_norm_(xLSTM_classifier.parameters(), max_norm=1.0)  # Apply gradient clipping to prevent the exploding gradient problem
            optimiser.step()


        # plot train logit and simoid distribution
        # all_logits = torch.cat(logits_list, dim=0)
        # plot_logits_distribution(all_logits, logits_type="raw") 
        # plot_logits_distribution(all_logits, logits_type="sigmoid")
        
        # Calculate training accuracy
        train_acc = correct_train / total_train * 100

        # Calculate F1 score for training
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        
        # Calculate confusion matrix
        train_conf_matrix = confusion_matrix(all_train_labels, all_train_preds)        
        
        # validate model
        xLSTM_classifier.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        
                
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move data to the selected device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred = xLSTM_classifier(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss

                # Calculate accuracy for validation
                predicted = (torch.sigmoid(pred) > 0.5).float()  # Threshold at 0.5 for binary classification
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
                
                all_val_preds.extend(predicted.cpu().numpy())  # Collecting predictions for F1 score
                all_val_labels.extend(batch_y.cpu().numpy())  # Collecting true labels


        # Calculate validation accuracy
        val_acc = correct_val / total_val * 100

        # Calculate F1 score for validation
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_val_labels, all_val_preds)

               
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print training and validation progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}, Validation Loss: {val_loss:.8f}, '
              f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%, '
              f'Train F1 Score: {train_f1:.4f}, Validation F1 Score: {val_f1:.4f}')
        
        print(f'train Confusion Matrix (Epoch {epoch + 1}):\n{train_conf_matrix}')   
        print(f'Val Confusion Matrix (Epoch {epoch + 1}):\n{conf_matrix}')
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(xLSTM_classifier.state_dict(), f'model/xlstm_classifier_{name}.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break            
            
    print("Training complete!")
    
if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the saved file.')
    parser.add_argument('--seq', default=100, type=int, help='The sequence length the model looks back on')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--config', default=str, help='yaml config path.')
    args = parser.parse_args()
    
    
    set_random_seed(args.seed)
    train_loader, val_loader, test_loader = create_dataloader(args.seq, args.config)
    
    xlstm_stack, input_projection, output_projection = create_xlstm_model(args.seq)
    model = xLSTMClassifier(input_projection, xlstm_stack, output_projection)
    train_model(model, train_loader, val_loader, args.name)
    
    
    # test model
    model.load_state_dict(torch.load(f'model/xlstm_classifier_{args.name}.pth'))
    model.to('cuda')
    model.eval()
    correct_test = 0
    total_test = 0
    all_test_preds = []
    all_test_labels = []
    # logits_list = []
    
    num = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            num+=1
            batch_x = batch_x.to('cuda')
            batch_y = batch_y.to('cuda')
            pred = model(batch_x)
            
            # log logits
            # logits_list.append(pred.cpu())
            
            predicted = (torch.sigmoid(pred) > 0.5).float()
            correct_test += (predicted == batch_y).sum().item()
            total_test += batch_y.size(0)
            
            all_test_preds.extend(predicted.cpu().numpy())  # Collecting predictions for F1 score
            all_test_labels.extend(batch_y.cpu().numpy())  # Collecting true labels
            
    test_acc = correct_test / total_test * 100
    # Calculate F1 score for validation
    test_f1 = f1_score(all_test_labels, all_test_preds, average='binary')
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_test_labels, all_test_preds)


    # Print training and validation progress
    print(f'Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}')
    print(f'Confusion Matrix :\n{conf_matrix}')  
    
    # plot test logits and sigmoid 
    # all_logits = torch.cat(logits_list, dim=0)
    # plot_logits_distribution(all_logits, logits_type="raw", mode='test') 
    # plot_logits_distribution(all_logits, logits_type="sigmoid", mode='test')
    print(f'num : {num}')