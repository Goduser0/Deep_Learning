import os
import pandas as pd

root_path = "My_TAOD/TA/TA_Results/CycleGAN"

subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

subfolders = [{'dataset':f.split(' ')[0], 'label':f.split(' ')[1], 'time':f.split(' ')[2],} for f in subfolders]

dataset_types = set(item['dataset'] for item in subfolders)

for i in ["30", "70", "200", "300", "400", "500", "1000", "1500", "2000"]:
    for item in dataset_types:
        dataset_save_path = f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}/{i}.csv"
        if not os.path.exists(f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}"):
            os.makedirs(f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}")
        df = pd.DataFrame()
        for subfolder in subfolders:
            if subfolder['dataset'] == item:
                img_csv = f"{root_path}/{subfolder['dataset']} {subfolder['label']} {subfolder['time']}/samples/3000epoch_{i}/generate_imgs.csv"
                img_csv = pd.read_csv(img_csv)
                df = pd.concat([df, img_csv])
        df.to_csv(dataset_save_path, index=False)
        
root_path = "My_TAOD/TA/TA_Results/ConGAN"

subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

subfolders = [{'dataset':f.split(' ')[0], 'label':f.split(' ')[1], 'time':f.split(' ')[2],} for f in subfolders]

dataset_types = set(item['dataset'] for item in subfolders)

for i in ["30", "70", "200", "300", "400", "500", "1000", "1500", "2000"]:
    for item in dataset_types:
        dataset_save_path = f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}/{i}.csv"
        if not os.path.exists(f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}"):
            os.makedirs(f"My_TAOD/Train_Classification/results/diff_num/{root_path.split('/')[-1]}/{item}")
        df = pd.DataFrame()
        for subfolder in subfolders:
            if subfolder['dataset'] == item:
                img_csv = f"{root_path}/{subfolder['dataset']} {subfolder['label']} {subfolder['time']}/samples/10000epoch_{i}/generate_imgs.csv"
                img_csv = pd.read_csv(img_csv)
                df = pd.concat([df, img_csv])
        df.to_csv(dataset_save_path, index=False)
