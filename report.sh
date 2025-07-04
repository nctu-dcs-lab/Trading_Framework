#!/bin/bash

for i in {1..10}
do
    echo python STB3_Trader_Test.py --model_name experiment_20240825_12 --seed "$i" > report/EXP12_"$i".txt
    python STB3_Trader_Test.py --model_name experiment_20240825_12 --seed "$i" > report/EXP12_"$i".txt
done

for i in {1..10}
do
    echo python STB3_Trader_Test.py --model_name experiment_20240820_11-1 --seed "$i" > report/EXP11-1_"$i".txt
    python STB3_Trader_Test.py --model_name experiment_20240820_11-1 --seed "$i" > report/EXP11-1_"$i".txt
done 

for i in {1..10}
do
    echo python STB3_Trader_Test.py --model_name experiment_20240820_11-2 --seed "$i" > report/EXP11-2_"$i".txt
    python STB3_Trader_Test.py --model_name experiment_20240820_11-2 --seed "$i" > report/EXP11-2_"$i".txt
done 