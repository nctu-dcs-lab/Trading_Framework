#!/bin/bash

for ma in 20;
do
    for trend_len in 76;
    do
        # echo python preprocess.py --save_dir synthetic_data -r --trend_len "$trend_len" --ma "$ma"
        # python preprocess.py --save_dir synthetic_data -r --trend_len "$trend_len" --ma "$ma"

        # echo python visualizeTrendCombine.py --trend_len "$trend_len" --ma "$ma" --data_path synthetic_data
        # python visualizeTrendCombine.py --trend_len "$trend_len" --ma "$ma" --data_path synthetic_data

        for prob in 50 60 70 80 90 100;
        do
            for seed in 1 2 3 4 5 6 7 8 9 10;
            do
                echo python STB3_CombineTest.py --seed "$seed" --env StockEnvCombineTrader --balance 500000 --prob "$prob" --trend_len "$trend_len" --ma "$ma" > CombinerTest/seed"$seed"/CombineTrader_trend_"$trend_len"_prob_"$prob"_ma_"$ma"_seed_"$seed"_Test_DailyMethod.txt 
                python STB3_CombineTest.py --seed "$seed" --env StockEnvCombineTrader --balance 500000 --prob "$prob" --trend_len "$trend_len" --ma "$ma" > CombinerTest/seed"$seed"/CombineTrader_trend_"$trend_len"_prob_"$prob"_ma_"$ma"_seed_"$seed"_Test_DailyMethod.txt 
            
                echo python visualizeReturnCombine.py --filename CombineTrader_trend_"$trend_len"_prob_"$prob"_ma_"$ma"_seed_"$seed"_DailyMethod --seed "$seed"
                python visualizeReturnCombine.py --filename CombineTrader_trend_"$trend_len"_prob_"$prob"_ma_"$ma"_seed_"$seed"_DailyMethod --seed "$seed"
            done
        done
    done
done

