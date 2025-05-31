ENV="/home/zhouquan/anaconda3/envs/DLstudy/bin/python My_TAOD/Train_Classification/main.py"

# $ENV --mode "train" --classification_net "VGG19_Pretrained" --dataset_class "PCB_200" --dataset_ratio "10-shot" --gpu_id "1" 
# $ENV --mode "train" --classification_net "VGG19_Pretrained" --dataset_class "PCB_Crop" --dataset_ratio "10-shot" --gpu_id "1" 
# $ENV --mode "train" --classification_net "VGG19_Pretrained" --dataset_class "DeepPCB_Crop" --dataset_ratio "10-shot" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/10.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/10.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/10.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/20.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/20.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/20.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/30.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/30.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/30.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/40.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/40.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/40.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/50.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/50.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/50.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/60.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/60.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/60.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/70.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/70.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/70.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/80.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/80.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/80.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/90.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/90.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/90.pth" --gpu_id "1" 

$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/DeepPCB_Crop 10-shot/models/100.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/100.pth" --gpu_id "1" 
$ENV --mode "test" --test_model_path "My_TAOD/Train_Classification/results/VGG19_FE/PCB_Crop 10-shot/models/100.pth" --gpu_id "1" 
