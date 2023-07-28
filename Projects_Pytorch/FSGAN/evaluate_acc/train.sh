# train 50, pre-trained model
for class in vit_base_patch16_224 resnet50;
do
{
# 1. ratio 1
python backbone.py \
    --gpu_ids 0 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix1 \
    --test_path ../dataset/NEU/NEU-50-r64/test \
    --n_epochs 50 \
    --log_name mix1_yespre \
    --batch_size 32 \
    --output results

# 2. ratio 2
python backbone.py \
    --gpu_ids 0 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix2 \
    --test_path ../dataset/NEU/NEU-50-r64/test \
    --n_epochs 50 \
    --log_name mix2_yespre \
    --batch_size 32 \
    --output results

# 3. ratio 3
python backbone.py \
    --gpu_ids 0 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix3 \
    --test_path ../dataset/NEU/NEU-50-r64/test \
    --n_epochs 50 \
    --log_name mix3_yespre \
    --batch_size 32 \
    --output results

# 4. ratio 4
python backbone.py \
    --gpu_ids 0 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-50-r64-pad/train_1500_mix4 \
    --test_path ../dataset/NEU/NEU-50-r64/test \
    --n_epochs 50 \
    --log_name mix4_yespre \
    --batch_size 32 \
    --output results
}
done


# train 30, pre-trained model
# for class in resnet18 resnet50 vit_base_patch16_224;
# do
# {
# # 1. ratio 1
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix1 \
#     --test_path ../dataset/NEU/NEU-30-r64/test \
#     --n_epochs 50 \
#     --log_name mix1_yespre \
#     --batch_size 32 \
#     --output results30

# # 2. ratio 2
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix2 \
#     --test_path ../dataset/NEU/NEU-30-r64/test \
#     --n_epochs 50 \
#     --log_name mix2_yespre \
#     --batch_size 32 \
#     --output results30

# # 3. ratio 3
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix3 \
#     --test_path ../dataset/NEU/NEU-30-r64/test \
#     --n_epochs 50 \
#     --log_name mix3_yespre \
#     --batch_size 32 \
#     --output results30

# # 4. ratio 4
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix4 \
#     --test_path ../dataset/NEU/NEU-30-r64/test \
#     --n_epochs 50 \
#     --log_name mix4_yespre \
#     --batch_size 32 \
#     --output results30
# }
# done

# # train 10, pre-trained model
# for class in resnet18 resnet50 vit_base_patch16_224;
# do
# {
# # 1. ratio 1
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-10-r64-pad/train_1500_mix1 \
#     --test_path ../dataset/NEU/NEU-10-r64/test \
#     --n_epochs 50 \
#     --log_name mix1_yespre \
#     --batch_size 32 \
#     --output results10

# # 2. ratio 2
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-10-r64-pad/train_1500_mix2 \
#     --test_path ../dataset/NEU/NEU-10-r64/test \
#     --n_epochs 50 \
#     --log_name mix2_yespre \
#     --batch_size 32 \
#     --output results10

# # 3. ratio 3
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-10-r64-pad/train_1500_mix3 \
#     --test_path ../dataset/NEU/NEU-10-r64/test \
#     --n_epochs 50 \
#     --log_name mix3_yespre \
#     --batch_size 32 \
#     --output results10

# # 4. ratio 4
# python backbone.py \
#     --gpu_ids 0 \
#     --model_name ${class} \
#     --pretrained True \
#     --train_path ../dataset/NEU/NEU-10-r64-pad/train_1500_mix4 \
#     --test_path ../dataset/NEU/NEU-10-r64/test \
#     --n_epochs 50 \
#     --log_name mix4_yespre \
#     --batch_size 32 \
#     --output results10
# }
# done