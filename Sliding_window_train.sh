for train_start_year in 2018 2019 2020 2021;
do 
    train_end_year=$((train_start_year + 2))
    test_start_year=$((train_start_year + 3))

    echo "preprocess.py -r --save_dir synthetic_data_dailyMethod --ma 20 --train_end $train_end_year"
    python preprocess.py -r --save_dir synthetic_data_dailyMethod --ma 20 --train_end "$train_end_year"

    echo "python STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name long_trader_${test_start_year}_step1 --mask --balance 100000 --start_year $train_start_year --end_year $train_end_year --train_start $train_start_year --train_end $train_end_year --test_start $test_start_year --test_end $test_start_year --env StockEnvOnlyOneShare-v5_2"
    python -W ignore STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name long_trader_"$test_start_year"_step1 --mask --balance 100000 --start_year "$train_start_year" --end_year "$train_end_year" --train_start "$train_start_year" --train_end "$train_end_year" --test_start "$test_start_year" --test_end "$test_start_year" --env StockEnvOnlyOneShare-v5_2 > train_log/long_trader_"$test_start_year"_step1.txt
    

    echo "python STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name long_trader_${test_start_year}_step2 --mask --balance 500000 --start_year $train_start_year --end_year $train_end_year --train_start $train_start_year --train_end $train_end_year --test_start $test_start_year --test_end $test_start_year --env StockEnvNoLimit-v3_2 --retrain --trained_model  long_trader_${test_start_year}_step1"
    python -W ignore STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name long_trader_"$test_start_year"_step2 --mask --balance 500000 --start_year "$train_start_year" --end_year "$train_end_year" --train_start "$train_start_year" --train_end "$train_end_year" --test_start "$test_start_year" --test_end "$test_start_year" --env StockEnvNoLimit-v3_2 --retrain --trained_model  long_trader_"$test_start_year"_step1 > train_log/long_trader_"$test_start_year"_step2.txt
   
    
    echo "python STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name short_trader_${test_start_year}_step1 --mask --balance 100000 --start_year $train_start_year --end_year $train_end_year --train_start $train_start_year --train_end $train_end_year --test_start $test_start_year --test_end $test_start_year --env StockEnvOnlyOneShare-v6_2"
    python -W ignore STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name short_trader_"$test_start_year"_step1 --mask --balance 100000 --start_year "$train_start_year" --end_year "$train_end_year" --train_start "$train_start_year" --train_end "$train_end_year" --test_start "$test_start_year" --test_end "$test_start_year" --env StockEnvOnlyOneShare-v6_2 > train_log/short_trader_"$test_start_year"_step1.txt
    

    echo "python STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name short_trader_${test_start_year}_step2 --mask --balance 500000 --start_year $train_start_year --end_year $train_end_year --train_start $train_start_year --train_end $train_end_year --test_start $test_start_year --test_end $test_start_year --env StockEnvNoLimit-v4_2 --retrain --trained_model  short_trader_${test_start_year}_step1"
    python -W ignore STB3_Trainer.py --lr 0.000003 --total_timesteps 10000000 --steps 2048 --save_name short_trader_"$test_start_year"_step2 --mask --balance 500000 --start_year "$train_start_year" --end_year "$train_end_year" --train_start "$train_start_year" --train_end "$train_end_year" --test_start "$test_start_year" --test_end "$test_start_year" --env StockEnvNoLimit-v4_2 --retrain --trained_model  short_trader_"$test_start_year"_step1 > train_log/short_trader_"$test_start_year"_step2.txt
    

done