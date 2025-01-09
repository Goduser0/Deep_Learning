import os

def get_subfolders(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

path = 'My_TAOD/TA/TA_Results/CycleGAN'
subfolders = get_subfolders(path)
print(subfolders)