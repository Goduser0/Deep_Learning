python_path="/home/zhouquan/anaconda3/envs/DLstudy/bin/python"
script_path="/home/zhouquan/MyDoc/Deep_Learning/My_PCB_det/metric.py"
cal_map_script="/home/zhouquan/MyDoc/Deep_Learning/My_PCB_det/cal_map_50_95.py"

cwd_path="My_PCB_det/Results/test" # ReWrite

for overlap in 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
do
    $python_path $script_path --cwd $cwd_path --MINOVERLAP $overlap
done

$python_path $cal_map_script --cwd $cwd_path

# python_path="/home/zhouquan/anaconda3/envs/DLstudy/bin/python"
# script_path="/home/zhouquan/MyDoc/Deep_Learning/My_PCB_det/metric.py"
# cal_map_script="/home/zhouquan/MyDoc/Deep_Learning/My_PCB_det/cal_map_50_95.py"

# cwd_path="My_PCB_det/Results/result1" # ReWrite

# $python_path $script_path --cwd $cwd_path

# $python_path $cal_map_script --cwd $cwd_path