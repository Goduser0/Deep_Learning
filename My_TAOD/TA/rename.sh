#! rm one model
root_path="My_TAOD/TA/TA_Results/WGAN_GP"
mapfile -t subfolders < <(find "$root_path" -mindepth 1 -maxdepth 1 -type d)
for folder in "${subfolders[@]}"; do
    dataset_path=$(sed -n 's/^ dataset_path:\t//p' "$folder/log.txt")   
    shotname=$(echo "$dataset_path" | rev | cut -d'/' -f3 | rev)

    last_part=$(basename "$folder")
    dataset=$(echo "$last_part" | cut -d' ' -f1)
    new_dataset="${dataset}[${shotname}]"
    new_last_part="${last_part/$dataset/$new_dataset}"
    new_path="${folder/$last_part/$new_last_part}"
    mv "$folder" "$new_path"
done