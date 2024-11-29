import os
os.chdir("../third_party/GANMetric/GAN-Metrics")

# make image txt list
real_path = "/home/user/duzongwei/Projects/FSGAN/dataset/NEU/NEU-300-r64/Cr"
fake_path = "/home/user/duzongwei/Projects/FSGAN/work_dir/generator/wgan-gp/Cr/epoch400"

real_list = "/home/user/duzongwei/Projects/FSGAN/work_dir/generator/real_path.txt"
os.system("ls -r " + real_path + '/' + "*.jpg" + '>' + real_list)

fake_list = "/home/user/duzongwei/Projects/FSGAN/work_dir/generator/fake_path.txt"
os.system("ls -r " + fake_path + '/' + "*.jpg" + '>' + fake_list)

# calculate JSD
os.system("python eval.py \
        --metric jsd \
        --pred_list /home/user/duzongwei/Projects/FSGAN/work_dir/generator/fake_path.txt \
        --gt_list /home/user/duzongwei/Projects/FSGAN/work_dir/generator/real_path.txt \
        --gpu_id 0 \
        --resize 128")