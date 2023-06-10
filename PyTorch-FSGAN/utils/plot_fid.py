import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd()) + "/metrics")
from matplotlib import pyplot as plt
import numpy as np
from metrics.FeatureNet import ConvNetFeatureExtract, fid


def save_fid(cls_path):
    """calculate fid with epoch and save it to `cls_path` as `fid_epoch.txt`. """
    feature_extract = ConvNetFeatureExtract(model="inception_v3", workers=4)
    epochs = [e for e in os.listdir(cls_path) if os.path.isdir(cls_path + '/' + e)]
    epochs = sorted(epochs, key=lambda x: int(x[5:]))
    # create fid_epoch.txt
    with open(os.path.join(cls_path, "fid_epoch.txt"), "a+") as log:
        for epoch in epochs:
            cls_ = cls_path.split('/')[-1]
            fake_path = os.path.join(cls_path, epoch)
            real_path = "../dataset/NEU/NEU-50-r64/train/" + cls_
            real_feature = feature_extract.extractFeature(real_path)
            fake_feature = feature_extract.extractFeature(fake_path)
            fid_epoch = fid(real_feature, fake_feature)

            # write fid_epoch.txt log
            item = "%s fid:%f" % (epoch, fid_epoch)
            log.write(item + '\n')


def plot_fid(log_path, inter=1, cls_type='Cr'):
    random_sample_init_fid = {'Cr':15.7, 'In':20.4, 'Pa':17.7, 'PS':15.7, 'RS':18.8, 'Sc':14.9}
    plt.figure(figsize=(10, 4), dpi=150)
    with open(log_path, 'r') as log:
        lines = log.readlines()
        n = len(lines)
        epoch, fid = np.zeros(n, dtype=np.int32), np.zeros(n)
        for i, line in enumerate(lines):
            item_list = line.split()
            epoch[i] = int(item_list[0][5:])
            fid[i] = float(item_list[1][4:])

        # plt fid
        epoch = epoch[::inter]
        fid = fid[::inter]
        epoch = np.insert(epoch, 0, 0)
        fid = np.insert(fid, 0, random_sample_init_fid[cls_type])
        x = list(range(len(epoch)))
        plt.title(cls_type)
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.xticks(x, epoch, rotation=45, fontsize=8)
        plt.plot(x, fid, label='FID')
        plt.legend(loc=1)
        fig_path = os.path.dirname(os.path.abspath(log_path)) + "/fid_epoch.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()  # release figure memory 


if __name__ == '__main__':
    # ----------------------------
    # write fid_epoch.txt
    # ----------------------------
    # root = "../work_dir/generator"
    # gan_type = [root + '/' + gt for gt in os.listdir(root) if os.path.isdir(root + '/' + gt)]
    # for gt in gan_type:
    #     for cls_ in os.listdir(gt):
    #         cls_path = gt + '/' + cls_
    #         save_fid(cls_path)

    # save_fid("../work_dir/generator/sphere/In")
    # save_fid("../work_dir/generator/sphere/Pa")

    # ----------------------------
    # plot fid_epoch.png
    # ----------------------------
    # plot_fid("../work_dir/generator/dragan/Cr/fid_epoch.txt")
    root = "../work_dir/generator/"
    gan_type = [root + gt for gt in os.listdir(root) if os.path.isdir(root + gt)]
    for gt in gan_type:
        for cls_ in os.listdir(gt):
            log_path = gt + '/' + cls_ + "/fid_epoch.txt"
            plot_fid(log_path, cls_type=cls_)
