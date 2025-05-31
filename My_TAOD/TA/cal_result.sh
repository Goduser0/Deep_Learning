#! rm one model
# root_path="My_TAOD/TA/TA_Results/CoGAN"
# root_path="My_TAOD/TA/TA_Results/CycleGAN"
# mapfile -t subfolders < <(find "$root_path" -mindepth 1 -maxdepth 1 -type d)
# for folder in "${subfolders[@]}"; do
#   rm -rf "$folder/samples"/*
# done

#! rm all
# root_path="My_TAOD/TA/TA_Results"
# mapfile -t subfolders < <(find "$root_path" -mindepth 2 -maxdepth 2 -type d)
# for folder in "${subfolders[@]}"; do
#   rm -rf "$folder/samples"/*
# done

#! cal_result all
root_path="My_TAOD/TA/TA_Models"
ENV="/home/zhouquan/anaconda3/envs/DLstudy/bin/python"

find "$root_path" -mindepth 2 -maxdepth 2 -type f -name "generator.py" | while IFS= read -r file; do
  $ENV $file
done
