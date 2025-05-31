ENV1="/home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/train_test_filter.py"
ENV2="/home/zhouquan/anaconda3/envs/DLstudy/bin/python filter.py"

root_path="My_TAOD/dataset/filter"
mapfile -t gans < <(find "$root_path" -mindepth 1 -maxdepth 1 -type d)

# for gan in "${gans[@]}"; do
#     mapfile -t datasets < <(find "$gan" -mindepth 1 -maxdepth 1 -type d)
    
#     for dataset in "${datasets[@]}"; do
#         mapfile -t shots < <(find "$dataset" -mindepth 1 -maxdepth 1)       
#         for SHOT in "${shots[@]}"; do
#             echo $SHOT
#             # $ENV --add_train_csv "$arg"
#         done
#     done
# done

# $ENV --add_train_csv "My_TAOD/dataset/filter/CycleGAN/PCB_200[10-shot]<-DeepPCB_Crop[200-shot]/1500.csv"
# $ENV --add_train_csv "My_TAOD/dataset/filter/CycleGAN/PCB_200[10-shot]<-DeepPCB_Crop[200-shot]/1500_renn.csv"
# $ENV --add_train_csv "My_TAOD/dataset/filter/CycleGAN/PCB_200[10-shot]<-DeepPCB_Crop[200-shot]/1500_renn_iht.csv"

# ks=(10 15 20 25 30 35 40 45 50 55 60)
# rs=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)

ks=(35)
rs=(0.2)

# $ENV1 --add_train_csv "My_TAOD/dataset/filter/ConGAN/PCB_200[10-shot]/1500.csv" --save_dir "My_TAOD/Train_Classification/results/filter/ConGAN/PCB_200[10-shot]/Resnet18_Pretrained/base"
for k in "${ks[@]}"; do
    for r in "${rs[@]}"; do
        # $ENV2 --k $k --r $r
        $ENV1 --add_train_csv "My_TAOD/dataset/filter/ConGAN/PCB_200[10-shot]/1500_renn.csv" --save_dir "My_TAOD/Train_Classification/results/filter/ablation/renn"
        $ENV1 --add_train_csv "My_TAOD/dataset/filter/ConGAN/PCB_200[10-shot]/1500_renn_iht.csv" --save_dir "My_TAOD/Train_Classification/results/filter/ablation/renn_iht"
        $ENV1 --add_train_csv "My_TAOD/dataset/filter/ConGAN/PCB_200[10-shot]/1500_renn_iht_rnn.csv" --save_dir "My_TAOD/Train_Classification/results/filter/ablation/renn_iht_rnn"
    done
done