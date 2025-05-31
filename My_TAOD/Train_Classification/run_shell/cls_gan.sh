MODELS=("Resnet18_Pretrained" "Resnet50_Pretrained" "EfficientNet_Pretrained" "VGG11_Pretrained" "MobileNet_Pretrained")
ENV="/home/zhouquan/anaconda3/envs/DLstudy/bin/python /home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/train_test.py"

root_path="My_TAOD/TA/TA_Results"
mapfile -t gans < <(find "$root_path" -mindepth 1 -maxdepth 1 -type d)

for gan in "${gans[@]}"; do
    mapfile -t datasets < <(find "$gan" -maxdepth 1 -type f -name "*.csv")
    for dataset in "${datasets[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            $ENV --add_train_csv "[\"$dataset\"]" --classification_net "$MODEL"
        done
    done
done
