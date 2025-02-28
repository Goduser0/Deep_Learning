import os

def get_subfolders(path):
    return [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

path = 'My_TAOD/TA/TA_Results/ConGAN'
subfolders = get_subfolders(path)
print(subfolders)