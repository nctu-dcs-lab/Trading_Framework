import gymnasium as gym
import StockEnv
from StockEnv.envs import StockEnvV2
import numpy as np
import torch
import random
import pandas as pd
def get_curr_time(df, i):
    curr_date = df.loc[i]["date"].date()
    curr_time = df.loc[i]["date"].strftime('%H:%M:%S')
    
    return curr_date, curr_time


def get_initial_contract_price(start_year):
    if start_year == 2021:
        contract_price = 33250
        initial_margin = 33250
        maintenance_margin = 25500
    elif start_year == 2022 or start_year == 2023:
        contract_price = 46000
        initial_margin = 46000
        maintenance_margin = 35250
    elif start_year == 2024:
        contract_price = 41750
        initial_margin = 41750
        maintenance_margin = 32000    

    return contract_price, initial_margin, maintenance_margin

def update_contract_price(date, original_price, original_initial_margin, original_maintenance_margin):
    contract_price = original_price
    initial_margin = original_initial_margin
    maintenance_margin = original_maintenance_margin

    if(date == pd.Timestamp("2020-11-13 13:30:00")):
        contract_price = 33250
        initial_margin = 33250
        maintenance_margin = 25500
    elif(date == pd.Timestamp("2021-01-06 13:30:00")):
        contract_price = 37500
        initial_margin = 37500
        maintenance_margin = 28750
    elif(date == pd.Timestamp("2021-02-05 13:30:00")):
        contract_price = 41750
        initial_margin = 41750
        maintenance_margin = 32000
    elif(date == pd.Timestamp("2021-05-19 13:30:00")):
        contract_price = 46000
        initial_margin = 46000
        maintenance_margin = 35250
    elif(date == pd.Timestamp("2022-01-26 13:30:00")):
        contract_price = 50750
        initial_margin = 50750
        maintenance_margin = 39000
    elif(date == pd.Timestamp("2022-02-08 13:30:00")):
        contract_price = 46000
        initial_margin = 46000
        maintenance_margin = 35250
    elif(date == pd.Timestamp("2023-01-17 13:30:00")):
        contract_price = 50750
        initial_margin = 50750
        maintenance_margin = 39000
    elif(date == pd.Timestamp("2023-01-31 13:30:00")):
        contract_price = 46000
        initial_margin = 46000
        maintenance_margin = 35250
    elif(date == pd.Timestamp("2023-07-28 13:30:00")):
        contract_price = 41750
        initial_margin = 41750
        maintenance_margin = 32000
    elif(date == pd.Timestamp("2024-02-05 13:30:00")):
        contract_price = 46000
        initial_margin = 46000
        maintenance_margin = 35250
    elif(date == pd.Timestamp("2024-02-16 13:30:00")):
        contract_price = 41750
        initial_margin = 41750
        maintenance_margin = 32000
    elif(date == pd.Timestamp("2024-03-07 13:30:00")):
        contract_price = 44750
        initial_margin = 44750
        maintenance_margin = 34250
    elif(date == pd.Timestamp("2024-05-03 13:30:00")):
        contract_price = 49500
        initial_margin = 49500
        maintenance_margin = 38000
    elif(date == pd.Timestamp("2024-05-17 13:30:00")):
        contract_price = 54500
        initial_margin = 54500
        maintenance_margin = 41750
    elif(date == pd.Timestamp("2024-06-25 13:30:00")):
        contract_price = 60250
        initial_margin = 60250
        maintenance_margin = 46250
    elif(date == pd.Timestamp("2024-08-09 13:30:00")):
        contract_price = 66250
        initial_margin = 66250
        maintenance_margin = 50750
    elif(date == pd.Timestamp("2024-08-22 13:30:00")):
        contract_price = 73000
        initial_margin = 73000
        maintenance_margin = 56000
    elif(date == pd.Timestamp("2024-09-27 13:30:00")):
        contract_price = 80500
        initial_margin = 80500
        maintenance_margin = 61750
    elif(date == pd.Timestamp("2024-11-13 13:30:00")):
        contract_price = 80500
        initial_margin = 80500
        maintenance_margin = 61750
        
    return contract_price, initial_margin, maintenance_margin

def calculate_share_value(contract_price, close, bid_price, share_num, direction):
    value = 0
    if direction == 1:
        value += contract_price * share_num
        value -= (close * 50 * 0.00002) * share_num
        value += ((close - bid_price) * 50) * share_num        
    else:
        value += contract_price * share_num
        value -= (close * 50 * 0.00002) * share_num
        value += ((bid_price - close) * 50) * share_num           

    return value

start_year = 2021
end_year = 2024
total_cost = 500000
total_balance = 500000
contract_price, initial_margin, maintenance_margin = get_initial_contract_price(start_year)
share_num = 0
available_num = 0

RoR = [0] * (end_year + 1 - start_year) * 12
month_count  = 0
for year in range(start_year, end_year+1):
    for month in range(1, 13):
        # load minute data
        minute_data = pd.read_csv(f'MTX_15Min_Data_Preprocessed/tfe-mtx00-{year}{month:02d}-15min.csv')
        
        # Convert the 'date' column to datetime objects (from strings or other formats)
        minute_data['date'] = pd.to_datetime(minute_data['date'])

        # initial setting
        curr_direction = 0
        balance = 500000
        bid_price = 0
        
        for i in range(len(minute_data)):
            curr_date, curr_time  = get_curr_time(minute_data, i)
            # update contract price and margin
            contract_price, initial_margin, maintenance_margin = update_contract_price(minute_data.loc[i]["date"], contract_price, initial_margin, maintenance_margin)
            print(f'share num : {share_num}')
            if i == 0:
                curr_direction = 1
                transaction_fee = minute_data.iloc[i]['close'] * 50 * 0.00002
                available_num = int((balance*0.8) // (contract_price+transaction_fee))
                bid_price = minute_data.iloc[i]['close']
                balance -= (contract_price + transaction_fee) * available_num
                share_num += available_num
                
            if i == len(minute_data)-1:
                balance += calculate_share_value(contract_price, minute_data.iloc[i]['close'], bid_price, share_num, curr_direction)
                share_num = 0 
                profit = balance - 500000
                total_balance += profit
                if total_balance < 500000:
                    total_cost += 500000 - total_balance
                    total_balance = 500000   
                RoR[month_count] = profit/500000
                month_count += 1          
                print(f'direction : {curr_direction}')
                print(f'bid price : {bid_price}')
                print(f'close : {minute_data.iloc[i]["close"]}')
                print(f'end month profit : {profit}')  
                print(f'end month RoR : {profit/500000}')
                print(f'---------'*5)                 
                
            # check balance is fit the margin
            broke = 0
            asset = 0
            if curr_direction == 1:
                asset = balance + (contract_price + (50 * (minute_data.iloc[i]['close'] - bid_price))) * share_num
                if asset < maintenance_margin * share_num:
                    broke = 1

                            
            if broke:
                if curr_direction == 1:
                    balance += calculate_share_value(contract_price, minute_data.iloc[i]['close'], bid_price, share_num, curr_direction)
                print(f'balance : {balance}')
                print(f'maintenance_margin : {maintenance_margin*share_num}')
                print(f'balance under margin at {curr_date} {curr_time}')


                profit = balance - 500000
                total_balance += profit
                if total_balance < 500000:
                    total_cost += 500000 - total_balance
                    total_balance = 500000    
                share_num = 0   
                print(f'total cost : {total_cost}')
                print(f'total balance : {total_balance}') 
                print(f'direction : {curr_direction}')
                print(f'bid price : {bid_price}')
                print(f'close : {minute_data.iloc[i]["close"]}')
                print(f'end month profit : {profit}')  
                print(f'end month RoR : {profit/500000}')
                print(f'---------'*5)     
                break  

            


print(f'total cost : {total_cost}')
print(f'total balance : {total_balance}')
ROI = (total_balance-total_cost)/total_cost
IRR = (1+ROI) ** (1/4) - 1
AVOL = np.std(RoR) * (12 ** 0.5)
print(f'ROI : {ROI}')
print(f'IRR : {IRR}')
print(f'Ann Vol : {AVOL}')
print(f'Sharp ratio : {(IRR - 0.017) / AVOL}')
