import random
import os
import shutil
import argparse
from unittest import main

parse = argparse.ArgumentParser()
parse.add_argument("--image_dir", type=str, default="../dataset/NEU/NEU-50-r64-pad")
parse.add_argument("--mix_ratio", type=float, default= 1 / 1, help="sys : pad = a / b")
parse.add_argument("--classic", type=str, default="geo-color/train_1500")
parse.add_argument("--synthesis", type=str, default="wgan-gp-my")
parse.add_argument("--mix", type=str, default="train_1500_mix")
config = parse.parse_args()

G_img_path = config.image_dir + '/' + config.synthesis
P_img_path = config.image_dir + '/' + config.classic
Mix_ratio = config.mix_ratio
Mix_img_path = config.image_dir + '/' + config.mix + "%.0f" % Mix_ratio
cls_list= os.listdir(P_img_path)
train_num = len(os.listdir(P_img_path + '/Cr'))



def offline_mix(syn_path, pad_path, mix_path):
    """mix imgaes according to mix ratio.
    Args:
        syn_path (_type_): the image path for Generator synthesis images.
        pad_path (_type_): the image path for classic transform images.
        mix_path (_type_): mix image path.
    """
    os.makedirs(mix_path)
    P_img_num = int(train_num / (1 + Mix_ratio))
    G_img_num = int(train_num - P_img_num)
    for cls in cls_list:
        t_path = os.path.join(mix_path, cls)
        os.makedirs(t_path, exist_ok=True)
        g_path, p_path = os.path.join(syn_path, cls), os.path.join(pad_path, cls)
        g_path_dir, p_path_dir = os.listdir(g_path), os.listdir(p_path)

        g_sample = random.sample(g_path_dir, G_img_num)
        for g_img in g_sample:
            shutil.copy(g_path + '/' + g_img, t_path + '/' + g_img)

        p_sample = random.sample(p_path_dir, P_img_num)
        for p_img in p_sample:
            shutil.copy(p_path + '/' + p_img, t_path + '/' + p_img)

        print('------------------- ' + cls + ' have been mixed -------------------')


def check(path):
    import re
    num = 0
    for i in os.listdir(path):
        if re.search("pad", i):
            num += 1
    print(num)
    
    
if __name__ == '__main__':
    offline_mix(G_img_path, P_img_path, Mix_img_path)
    # check(Mix_img_path + "/Cr")