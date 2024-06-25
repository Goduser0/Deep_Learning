import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cwd', type=str, default='My_PCB_det/Results/test')
config = parser.parse_args()
out_file_path = os.path.join(config.cwd, 'output')
map_result_list = [i for i in os.listdir(out_file_path) if os.path.isdir(os.path.join(out_file_path, i))]

mAP = []
for map_result in map_result_list:
    with open(out_file_path + '/'+ map_result + "/output.txt", 'r') as file:
        for line in file:
            if "mAP@" in line:
                mAP.append(float(line.strip().split(' ')[-1][:-1]))
                
with open(out_file_path + "/output.txt", 'w') as file:
    file.write(f"mAP@.5:.95 = {sum(mAP)/len(mAP)}%")
file.close()
print(f"mAP@.5:.95 = {sum(mAP)/len(mAP)}%")