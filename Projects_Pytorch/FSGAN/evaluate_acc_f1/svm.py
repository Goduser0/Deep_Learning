import numpy as np  #用于数据处理
from matplotlib import pyplot as plt  #用于显示图像和画图
from sklearn import svm #导入支持向量机
from sklearn.model_selection  import train_test_split #用于数据集划分
from sklearn.metrics import accuracy_score  #用于计算正确率
import cv2  #用于读取图片
import os  #文件读取
import pickle  #用于模型的保存
from time import time
from sklearn.metrics import accuracy_score, f1_score


SHAPE = (64, 64) #设置输入图片的大小


def extractFeaturesFromImage(image_file):
    img = cv2.imread(image_file)#读取图片
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    #对图片进行risize操作统一大小
    img = img.flatten()#对图像进行降维操作，方便算法计算, (64, 64, 3) -> 12288
    img = img / np.mean(img)#归一化，突出特征
    return img


def getImageData(directory):
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            for image in images:
                s += 1
                label_list.append(d)
                feature_list.append(extractFeaturesFromImage(root + d + "/" + image))

    return np.asarray(feature_list), np.asarray(label_list)


def train(dir, i):
    #数据获取，这里Svm_derection是自定义类的名称
    feature_array, label_array = getImageData(dir)
    #数据的分割
    X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

    print("shape of raw image data: {0}".format(feature_array.shape))
    print("shape of raw image data: {0}".format(X_train.shape))
    print("shape of raw image data: {0}".format(X_test.shape))
    #模型的选择
    clf = svm.SVC(gamma=0.001, C=100., probability=True)
    #模型的训练
    clf.fit(X_train, y_train);
    #模型测试
    Ypred = clf.predict(X_test);

    print("pre",Ypred)
    print("test",y_test)
    #模型保存
    pickle.dump(clf, open(f"svm_{i}.pkl", "wb"))


def test(img_file, clf):
    Ypred = clf.predict(np.reshape(extractFeaturesFromImage(img_file), (1, -1)))
    return Ypred[0]


if __name__ == "__main__":
    for i in range(5):
        # Train
        train("/home/duzongwei/Projects/FSGAN/dataset/SDPCB/PCB-50-r64/train/", i)
        print("Train Complished!!!")

        # Test
        pkl_file = open(f"svm_{i}.pkl", 'rb')
        clf = pickle.load(pkl_file)

        true_list = []
        pred_list = []
        start_time = time()
        for root, dirs, files in os.walk("/home/duzongwei/Projects/FSGAN/dataset/SDPCB/PCB-50-r64/test/"):
            for i in files:
                test_img_file = os.path.join(root, i)
                true = test_img_file.split('/')[-2]
                pred = test(test_img_file, clf)
                true_list.append(true)
                pred_list.append(pred)
        end_time = time()
        print(accuracy_score(true_list, pred_list), f1_score(true_list, pred_list, average='macro'))
        print("All time cost: ", end_time - start_time)