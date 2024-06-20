import argparse
import os
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
import json
import warnings
from utils import txt2box, compute_ap, draw_text_in_image, log_average_miss_rate, draw_plot_func
from PIL import Image

warnings.filterwarnings('ignore')

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

#####################################################################################################################################
## CONFIG
#####################################################################################################################################
parser = argparse.ArgumentParser()
# parser.add_argument('--cwd', type=str, help="current work dictionary", default='My_PCB_det/mAP-master')
parser.add_argument('--cwd', type=str, help="current work dictionary", default='My_PCB_det/result')
parser.add_argument('--dataset', type=str)
parser.add_argument('--animation', help="animation is shown.", type=bool, default=True)
parser.add_argument('--plot', help="plot is shown", type=bool, default=True)
parser.add_argument('--SaveTempFile', type=bool)
parser.add_argument('--MINOVERLAP', type=float, default=0.5)
config = parser.parse_args()

config.dataset = 'tzb'
config.SaveTempFile = False

MINOVERLAP = config.MINOVERLAP
GT_PATH = os.path.join(config.cwd, 'input', 'ground-truth')
DR_PATH = os.path.join(config.cwd, 'input', 'detection-results')
IMG_PATH = os.path.join(config.cwd, 'input', 'images-optional')
OUTPUT_PATH = os.path.join(config.cwd, 'output', 'map_'+str(int(MINOVERLAP*100)))
TEMP_FILES_PATH = os.path.join(config.cwd, ".temp_files")

if os.path.exists(TEMP_FILES_PATH):
    if config.SaveTempFile == False:
        shutil.rmtree(TEMP_FILES_PATH)
        os.makedirs(TEMP_FILES_PATH)
else:
    os.makedirs(TEMP_FILES_PATH)
    
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

if config.animation:
    os.makedirs(os.path.join(OUTPUT_PATH, "images", "detections_one_by_one"))
if config.plot:
    os.makedirs(os.path.join(OUTPUT_PATH, "classes"))

if not os.path.exists(IMG_PATH):
    config.animation = False

ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
ground_truth_files_list.sort()

gt_counter_per_class = {} # 统计类别及数量
counter_images_per_class = {}
gt_files = []
for txt_file in ground_truth_files_list:
    file_id = txt_file.split('.txt', 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # 确保对应预测结果存在
    assert os.path.exists(os.path.join(DR_PATH, (file_id + ".txt")))
    # 确保对应图像结果存在
    assert (config.dataset not in ['tzb'] or os.path.exists(os.path.join(IMG_PATH, (file_id + ".bmp"))))
    lines_list = txt2box(txt_file, file_type='gt')
    # create ground-truth dictionary
    bounding_boxes = []
    already_seen_classes = []
    for line in lines_list:
        class_name, left, top, right, bottom = line
        if config.dataset in ['tzb']:
            img = Image.open(os.path.join(IMG_PATH, (file_id + ".bmp"))).convert('RGB')
            height, weight = np.array(img).shape[:2]
            left = (line[1] - 0.5*line[3])*weight
            right = (line[1] + 0.5*line[3])*weight
            top = (line[2] - 0.5*line[4])*height
            bottom = (line[2] + 0.5*line[4])*height
        
        bounding_boxes.append({"class_name": class_name, "bbox":[left, top, right, bottom], "used":False})
        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            gt_counter_per_class[class_name] = 1
            
        if class_name not in already_seen_classes:
            if class_name in counter_images_per_class:
                counter_images_per_class[class_name] += 1
            else:
                counter_images_per_class[class_name] = 1
            already_seen_classes.append(class_name)
            
    new_temp_file = TEMP_FILES_PATH + '/' + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w') as outfile:
        json.dump(bounding_boxes, outfile)

# 类别
gt_classes = list(gt_counter_per_class.keys())
gt_classes = sorted(gt_classes)
# 类别数
n_classes = len(gt_classes)


dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    # each class
    bounding_boxes = []
    for txt_file in dr_files_list:
        # each .txt file
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # 确保对应ground_turth结果存在
        assert os.path.exists(os.path.join(GT_PATH, (file_id + ".txt")))
        # 确保对应图像结果存在
        assert (config.dataset not in ['tzb'] or os.path.exists(os.path.join(IMG_PATH, (file_id + ".bmp"))))
        lines_list = txt2box(txt_file, file_type='dr')
        for line in lines_list:
            # each defect
            temp_class_name, confidence, left, top, right, bottom = line
            if config.dataset in ['tzb']:
                img = Image.open(os.path.join(IMG_PATH, (file_id + ".bmp"))).convert('RGB')
                height, weight = np.array(img).shape[:2]
                left = (line[2] - 0.5*line[4])*weight
                right = (line[2] + 0.5*line[4])*weight
                top = (line[3] - 0.5*line[5])*height
                bottom = (line[3] + 0.5*line[5])*height
            if temp_class_name == class_name:
                bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":[left, top, right, bottom]})
    bounding_boxes.sort(key=lambda x:(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + '/dr_' + class_name + '.json', 'w') as outfile:
        json.dump(bounding_boxes, outfile)

#####################################################################################################################################
## Calculate the AP for each class
#####################################################################################################################################
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
with open(OUTPUT_PATH + "/output.txt", 'w') as output_file:
    output_file.write("# AP and precision/recall pre class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        # each class
        count_true_positives[class_name] = 0
        # Load detection-results of that  class
        dr_file = TEMP_FILES_PATH + '/dr_' + class_name + '.json'
        dr_data = json.load(open(dr_file)) 
        # Assign detection-results to ground-truth objects
        nd = len(dr_data)
        tp = [0] * nd
        fp = [0] * nd
        
        for idx, detection in enumerate(dr_data):
            #each detection
            file_id = detection['file_id']
            
            if config.animation:
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                assert len(ground_truth_img) == 1
                # img = Image.open(IMG_PATH + '/' + ground_truth_img[0]).convert('RGB')
                img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                img_output_path = OUTPUT_PATH + '/images/' + ground_truth_img[0]
                if os.path.isfile(img_output_path):
                    img_cumulative = cv2.imread(img_output_path)
                else:
                    img_cumulative = img.copy()
                # Add bottom border to image
                bottom_border = 60
                BLACK = [0, 0, 0]
                img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            # 如果有与该 file_id 相同的开放真实数据，则将检测结果分配给真实数据对象
            gt_file = TEMP_FILES_PATH + '/' + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = detection["bbox"]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = obj["bbox"]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
                            
            if config.animation:
                status = "NO MATCH FOUND!"
            
            if ovmax >= MINOVERLAP:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                    if config.animation:
                        status = "MATCH!"
                else:
                    # false positive
                    fp[idx] = 1
                    if config.animation:
                        status = "REPEATED MATCH!"
            else:
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"
        
            if config.animation:
                height, width = img.shape[:2]
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)

                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image:" + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]:" + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(MINOVERLAP*100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(MINOVERLAP*100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx+1) # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0: # if there is intersections between the bounding-boxes
                    bbgt = [int(round(float(x))) for x in gt_match["bbox"]]
                    cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # save image to output
                output_img_path = OUTPUT_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_output_path, img_cumulative)
        
        
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        
        ap, mrec, mprec = compute_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        
        # Write to output.txt
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec))
        lamr_dictionary[class_name] = lamr
        
        if config.plot:
            plt.plot(rec, prec, '-o')
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf() # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            #plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca() # gca - get current axes
            axes.set_xlim([0.0,1.0])
            axes.set_ylim([0.0,1.05]) # .05 to give some extra space
            # save the plot
            fig.savefig(OUTPUT_PATH + "/classes/" + class_name + ".png")
            plt.cla() # clear axes for next plot
    
    if config.animation:
        cv2.destroyAllWindows()
    
    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP@{0:d} = {1:.4f}%".format(int(MINOVERLAP*100), mAP*100)
    output_file.write(text + "\n")
    print(text)   

"""
 Draw false negatives
"""
if config.animation:
    pink = (203,192,255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        #print(ground_truth_data)
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = OUTPUT_PATH + "/images/" + img_id + ".bmp"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".bmp"
            img = cv2.imread(img_path)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
        cv2.imwrite(img_cumulative_path, img)
    
"""
 Count total of detection-results
"""
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = txt2box(txt_file, file_type='dr')
    for line in lines_list:
        class_name = line[0]
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
#print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())


# Plot the total number of occurences of each class in the ground-truth
if config.plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = OUTPUT_PATH + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )
    
# Write number of ground-truth objects per class to results.txt
with open(OUTPUT_PATH + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
        

# Finish counting true positives
for class_name in dr_classes:
    # if class exists in detection-result but not in ground-truth then there are no true positives in that class
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
        
"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if config.plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = OUTPUT_PATH + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )

"""
 Write number of detected objects per class to output.txt
"""
with open(OUTPUT_PATH + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)

"""
 Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
"""
if config.plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = OUTPUT_PATH + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if config.plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    x_label = "Average Precision"
    output_path = OUTPUT_PATH + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )