import matplotlib.pyplot as plt
import matplotlib.patches as plp
from PIL import Image
import numpy as np
import cv2
import math
import torch
import operator
import os
import re

# img_classes = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

#######################################################################################################
#### FUNCTION: box_corner2center()
#######################################################################################################
def box_corner2center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#######################################################################################################
#### FUNCTION: box_center2corner()
#######################################################################################################
def box_center2corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

#######################################################################################################
#### FUNCTION: txt2box()
#######################################################################################################
def txt2box(path, file_type='gt'):
    """
    txt文件形成目标框
    file_type = gt/dr
    mode = center/corner
    """
    locations = open(path, 'r')
    locations = locations.readlines()
    
    label_box = []
    for location in locations:
        if file_type == 'gt':
            label, x_cen, y_cen, box_w, box_h = [i for i in location.split(' ')]
            label_box.append([label, float(x_cen), float(y_cen), float(box_w), float(box_h)])
        elif file_type == 'dr':
            label, confidence, x_min, y_min, x_max, y_max = [i for i in location.split(' ')]
            label_box.append([label, float(confidence), float(x_min), float(y_min), float(x_max), float(y_max)])
    
    return label_box

#######################################################################################################
#### FUNCTION: box2rect()
#######################################################################################################
def box2rect(img_path, save_path, boxes, color='red'):
    img = Image.open(img_path).convert('RGB')
    height, weight = np.array(img).shape[:2]
    
    fig = plt.imshow(img)
    for box in boxes:
        fig.axes.add_patch(plp.Rectangle(
            xy=((box[1]-0.5*box[3])*weight, (box[2]-0.5*box[4])*height), 
            width=box[3]*weight, 
            height=box[4]*height, 
            fill=False, 
            edgecolor=color, 
            linewidth=0.2))
        fig.axes.text((box[1]-0.5*box[3])*weight, (box[2]-0.5*box[4])*height, str(int(box[0])), fontsize=2, color='black')
        
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"save to :{save_path}")

#######################################################################################################
#### FUNCTION: log_average_miss_rate()
#######################################################################################################
def log_average_miss_rate(prec, rec):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

#######################################################################################################
#### FUNCTION: is_float_between_0_and_1()
#######################################################################################################
def is_float_between_0_and_1(value):
    """
    check if the number is a float between 0.0 and 1.0 | (0.0, 1.0)
    """
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else: return False
    except ValueError:
        return False

#######################################################################################################
#### FUNCTION: compute_ap()
#######################################################################################################
def compute_ap(recall, precision, method='interp'):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # Integrate area under curve
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap, mrec, mpre

#######################################################################################################
#### FUNCTION: draw_text_in_image()
#######################################################################################################
def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

#######################################################################################################
#### FUNCTION: adjust_axes()
#######################################################################################################
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


#######################################################################################################
#### FUNCTION: draw_plot_func()
#######################################################################################################
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()

#######################################################################################################
#### FUNCTION: get_classes()
#######################################################################################################
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#######################################################################################################
#### FUNCTION: generate_anchor_base()
#######################################################################################################
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """生成基础的先验框"""
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#######################################################################################################
#### FUNCTION: _enumerate_shifted_anchor()
#######################################################################################################
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """对基础先验框进行拓展对应到所有特征点上"""
    #---------------------------------#
    #   计算网格中心点
    #---------------------------------#
    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    #---------------------------------#
    #   每个网格点上的9个先验框
    #---------------------------------#
    A       = anchor_base.shape[0]
    K       = shift.shape[0]
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

#######################################################################################################
#### FUNCTION: loc2bbox()
#######################################################################################################
def loc2bbox(src_bbox, loc):
    """src_bbox先验框, loc建议框结果"""
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    #计算先验框的宽、高，中心坐标
    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    #对先验框进行大小、坐标调整参数
    #[:,0::4]:所有行中，列下标为0,1,2，。。。改变其二维表格中的值。
    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    #先验框调整过程 
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox
#######################################################################################################
#### FUNCTION: cvtColor()
#######################################################################################################
def cvtColor(image):
    """
        将图像转换成RGB图像,防止灰度图在预测时报错。
        代码仅仅支持RGB图像的预测,所有其它类型的图像都会转化成RGB
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#######################################################################################################
#### FUNCTION: resize_image()
#######################################################################################################
def resize_image(image, size):
    """对输入图像进行resize"""
    w, h        = size
    new_image   = image.resize((w, h), Image.BICUBIC)
    return new_image

#######################################################################################################
#### FUNCTION: get_lr()
#######################################################################################################
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
#######################################################################################################
#### FUNCTION: preprocess_input()
#######################################################################################################
def preprocess_input(image):
    image /= 255.0
    return image

#######################################################################################################
#### FUNCTION: get_new_img_size()
#######################################################################################################
def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width

##########################################################################################################
# FUNCTION: logger
##########################################################################################################
def logger(config):
    assert os.path.exists(config.save_dir), f"ERROR:\t({__name__}): No config.save_dir"
    filename = config.time
    with open(config.save_dir + '/' + filename + '.txt', 'w') as f:
        for arg, value in vars(config).items():
            f.write(f"{arg}:\t{value}\n")
    f.close()
#######################################################################################################
#### Test
#######################################################################################################
if __name__ == '__main__':
    pass
    print(get_classes("My_Datasets/Detection/PCB_瑕疵初赛样例集/cls_classes.txt"))