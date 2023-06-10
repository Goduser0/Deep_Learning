# for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
# for class in resnet50 vit_base_patch16_224;
for class in resnet18;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 1 \
        --num_classes 5 \
        --model_name ${class} \
        --train_path ../dataset/SDPCB/PCB-50-r64-pad/train_1500_mix0.5 \
        --test_path ../dataset/SDPCB/PCB-50-r64/test \
        --n_epochs 50 \
        --log_name proposed_${i} \
        --batch_size 16 \
        --output results50_pcb_adam
    }&
    done
    wait
done

# for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
# for class in resnet50 vit_base_patch16_224;
for class in resnet18;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 1 \
        --num_classes 5 \
        --model_name ${class} \
        --train_path ../dataset/SDPCB/PCB-50-r64-pad/color \
        --test_path ../dataset/SDPCB/PCB-50-r64/test \
        --n_epochs 50 \
        --log_name color_${i} \
        --batch_size 16 \
        --output results50_pcb_adam
    }&
    done
    wait
done

# for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
# for class in resnet50 vit_base_patch16_224;
for class in resnet18;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 1 \
        --num_classes 5 \
        --model_name ${class} \
        --train_path ../dataset/SDPCB/PCB-50-r64-pad/geo-color \
        --test_path ../dataset/SDPCB/PCB-50-r64/test \
        --n_epochs 50 \
        --log_name geocolor_${i} \
        --batch_size 16 \
        --output results50_pcb_adam
    }&
    done
    wait
done

# for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
# for class in resnet50 vit_base_patch16_224;
for class in resnet18;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 1 \
        --num_classes 5 \
        --model_name ${class} \
        --train_path ../dataset/SDPCB/PCB-50-r64-pad/geometric \
        --test_path ../dataset/SDPCB/PCB-50-r64/test \
        --n_epochs 50 \
        --log_name geometric_${i} \
        --batch_size 16 \
        --output results50_pcb_adam
    }&
    done
    wait
done

# for class in resnet18 mobilenetv3_large_100 efficientnet_b0;
# for class in resnet50 vit_base_patch16_224;
for class in resnet18;
do
    for i in 1 2 3 4 5;
    do
    {
    python backbone.py \
        --gpu_ids 1 \
        --num_classes 5 \
        --model_name ${class} \
        --train_path ../dataset/SDPCB/PCB-50-r64/train \
        --test_path ../dataset/SDPCB/PCB-50-r64/test \
        --n_epochs 50 \
        --log_name noaug_${i} \
        --batch_size 16 \
        --output results50_pcb_adam
    }&
    done
    wait
done