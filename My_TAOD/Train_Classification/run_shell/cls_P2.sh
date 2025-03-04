#* single-domain

ENV="/home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py"

# train 30-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "Resnet18_Pretrained"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "Resnet50_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "EfficientNet_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "VGG11_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "MobileNet_Pretrained" 
# test 30-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "Resnet18_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet18_Pretrained 20241127_211008/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "Resnet50_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet50_Pretrained 20241127_211146/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "EfficientNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 EfficientNet_Pretrained 20241220_173546/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "VGG11_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 VGG11_Pretrained 20241220_184224/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "30-shot" --classification_net "MobileNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 MobileNet_Pretrained 20241220_184420/models/100.pth"

# train 10-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "Resnet18_Pretrained"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "Resnet50_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "EfficientNet_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "VGG11_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "MobileNet_Pretrained" 
# test 10-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "Resnet18_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet18_Pretrained 20241127_211809/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "Resnet50_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet50_Pretrained 20241127_211928/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "EfficientNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 EfficientNet_Pretrained 20241220_173931/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "VGG11_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 VGG11_Pretrained 20241220_184627/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "10-shot" --classification_net "MobileNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 MobileNet_Pretrained 20241220_184809/models/100.pth"

# train 160-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "Resnet18_Pretrained"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "Resnet50_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "EfficientNet_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "VGG11_Pretrained" 
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode train --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "MobileNet_Pretrained" 
# test 160-shot
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "Resnet18_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet18_Pretrained 20241225_154528/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "Resnet50_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 Resnet50_Pretrained 20241225_154820/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "EfficientNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 EfficientNet_Pretrained 20241225_155325/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "VGG11_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 VGG11_Pretrained 20241225_155942/models/100.pth"
# /home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py --mode test --dataset_class PCB_200 --gpu_id "1" --dataset_ratio "160-shot" --classification_net "MobileNet_Pretrained" --test_model_path "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/cls_single_domain/PCB_200 MobileNet_Pretrained 20241225_160513/models/100.pth"

# train 50-shot 70-shot 90-shot 110-shot 130-shot 150-shot
MODELS=("Resnet18_Pretrained" "Resnet50_Pretrained" "EfficientNet_Pretrained" "VGG11_Pretrained" "MobileNet_Pretrained")

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "50-shot" --classification_net "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "70-shot" --classification_net "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "90-shot" --classification_net "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "110-shot" --classification_net "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "130-shot" --classification_net "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    $ENV --mode "train" --dataset_class "PCB_200" --gpu_id "1" --dataset_ratio "150-shot" --classification_net "$MODEL"
done

#* test 50-shot
# $ENV --mode "test" --gpu_id "1" --test_model_path "My_TAOD/Train_Classification/results/PCB_200 EfficientNet_Pretrained 50-shot 20250304_150611/models/100.pth"