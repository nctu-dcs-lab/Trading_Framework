for year in 2020 2021 2022 2023 2024;
do
    for num in 1 2;
    do
        echo python train_xLSTM.py --name "$year"H"$num"  --config config/"$year"H"$num".yaml > train_log/xLSTM_"$year"H"$num".txt
        python train_xLSTM.py --name "$year"H"$num"  --config config/"$year"H"$num".yaml > train_log/xLSTM_"$year"H"$num".txt
    done
done