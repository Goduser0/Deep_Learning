## 统一的class_index
0-Mouse_bite
1-Open_circuit
2-Short
3-Spur
4-Spurious_copper

## Data_Preprocess
|-ToVOC.py # 将数据集转为自定义VOC数据集格式，只转换与赛题相同类别的；如果原始数据集中的annotation文件已经指定class_index，此步也应将class_index转换成与赛题数据集一致的class_index
|   |-VOC数据集格式应为
|       |-Annotations # 原始annotation
|       |-ImageSets # 原始图片
|       |-cls_classes.txt
|
|-annotation.py # 完成数据集划分，完成dataset标准格式的转换；原始annotation中没有指定class_index的，指定为一致的class_index，保存在.txt文件中
|   |-VOC数据集格式应为
|   |   |-Annotations
|   |   |-ImageSets
|   |   |-cls_classes.txt
|   |   |-test.txt # 测试集
|   |   |-trainval.txt # 训练集
|   |-dataset.txt格式应为，每行组成为：[img_path [left, top, right, bottom, class_index] [obj_2] ...]，其中坐标全为整数像素坐标
|
|-txt2result.py # 将VOC数据集中的test.txt文件抽取为计算map所需的文件格式 
    |-计算map所需文件格式
        |-input
            |-ground-truth
            |-images-optional