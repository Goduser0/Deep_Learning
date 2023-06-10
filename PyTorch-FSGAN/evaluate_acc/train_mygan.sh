# small
for class in mobilenetv3_large_100;
do
{
python backbone_light.py \
    --gpu_ids 1 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix0.5 \
    --test_path ../dataset/NEU/NEU-30-r64/test \
    --n_epochs 50 \
    --log_name mix0.5_yespre \
    --batch_size 32 \
    --output results30

python backbone_light.py \
    --gpu_ids 1 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix0.5 \
    --test_path ../dataset/NEU/NEU-30-r64/test \
    --n_epochs 50 \
    --log_name mix0.5_yespre \
    --batch_size 32 \
    --output results30

python backbone_light.py \
    --gpu_ids 1 \
    --model_name ${class} \
    --pretrained True \
    --train_path ../dataset/NEU/NEU-30-r64-pad/train_1500_mix0.5 \
    --test_path ../dataset/NEU/NEU-30-r64/test \
    --n_epochs 50 \
    --log_name mix0.5_yespre \
    --batch_size 32 \
    --output results30
}
done