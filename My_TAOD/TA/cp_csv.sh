root_path="My_TAOD/Train_Classification/results/diff_num"

mapfile -t subfolders1 < <(find "$root_path" -mindepth 4 -maxdepth 4 -type d)

for folder1 in "${subfolders1[@]}"; do
    src=$(echo "$folder1/all_train.csv")
    gan=$(echo "$src" | cut -d'/' -f5)
    shotname=$(echo "$src" | cut -d'/' -f6)
    num_name=$(echo "$src" | cut -d'/' -f8)
    
    new_path=$(echo "My_TAOD/dataset/filter/$gan/$shotname/$num_name.csv")
    mkdir -p "$(dirname "$new_path")"
    cp "$src" "$new_path"
done