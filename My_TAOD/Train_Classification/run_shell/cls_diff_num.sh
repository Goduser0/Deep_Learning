ENV="/home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/train_test_diff_num.py"
SHOTS=("30.csv" "70.csv" "200.csv" "300.csv" "400.csv" "500.csv" "1000.csv" "1500.csv" "2000.csv")

root_path="My_TAOD/dataset/diff_num"
mapfile -t gans < <(find "$root_path" -mindepth 1 -maxdepth 1 -type d)

for gan in "${gans[@]}"; do
    mapfile -t datasets < <(find "$gan" -mindepth 1 -maxdepth 1 -type d)
    
    for dataset in "${datasets[@]}"; do
        csv_list=()        
        for SHOT in "${SHOTS[@]}"; do
            csv_list+=("\"${dataset}/${SHOT}\"")
            arg=$(echo "[${csv_list[*]}]" | sed 's/ /,/g')
            $ENV --add_train_csv "$arg"
        done
    done
done
