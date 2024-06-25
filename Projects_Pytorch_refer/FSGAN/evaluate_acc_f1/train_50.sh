# 1. No Aug
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64/train \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name noaug_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 2. Color
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/color \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name color_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 3. geo
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/geometric \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name geometric_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 4. color-geo
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/geo-color \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name geometric_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 5. wgan-lp
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix0.5_wgan-lp \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name mix0.5_wganlp_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 6. sagan
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix0.5_wgan-gp \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name mix0.5_sagan_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 7. hinge
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix0.5_hinge \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name mix0.5_hinge_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done

# 8. proposed
for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 0 \
        --model_name ${class} \
        --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix0.5 \
        --test_path ../dataset/NEU/NEU-50-r64/test \
        --n_epochs 50 \
        --log_name mix0.5_proposed_${i} \
        --batch_size 16 \
        --output results50
    }&
    done
    wait
done