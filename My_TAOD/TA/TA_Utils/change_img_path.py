import os
import pandas as pd

root_path = "My_TAOD/TA/TA_Results/WGAN_GP"

subfolders = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

for subfolder in subfolders:
    add_content = f"[{subfolder.split('/')[-1].split(']')[0].split('[')[-1]}]"
    sample_paths = [os.path.join(f"{subfolder}/samples", f) for f in os.listdir(f"{subfolder}/samples") if os.path.isdir(os.path.join(f"{subfolder}/samples", f))]
    for sample_path in sample_paths:
        img_csv = f"{sample_path}/generate_imgs.csv"
        df = pd.read_csv(img_csv)
        df['Image_Path'] = df['Image_Path'].astype(str).apply(lambda x: ' '.join([x.split(' ')[0] + add_content] + x.split(' ')[1:]))
        df.to_csv(img_csv, index=False)
