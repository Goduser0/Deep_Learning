import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
import palettable
import random

iris=datasets.load_iris()
data,kind=iris.data,iris.target
feature=np.asarray(iris.feature_names)
df_iris=pd.DataFrame(np.hstack((data,kind.reshape(150,1))),columns=np.append(feature,'class'))

plt.figure(dpi=150)
data_hist=df_iris[feature[0]]
n, bins, patches=plt.hist(x=data_hist,
                          
                          #箱子数(bins)设置，以下三种不能同时并存
                          bins=10,#default:10
                          #bins=[4,6,8],#分两个箱子，边界分别为[4,6),[6,8]
                          #bins='auto',# 可选'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                         
                          #range=(5,7),#数据范围，不指定时，为(x.min(), x.max())
                          #density=True, #默认为False，y轴显示频数；为True y轴显示频率
                          #weights=np.random.rand(len(x)),#对x中每一个样本设置权重，这里随机设置了权重
                          cumulative=False,#默认False，是否累加频数或者频率
                          bottom=0,#设置箱子y轴方向基线，默认为0
                          histtype='bar',#直方图类型默认为bar{'bar', 'barstacked', 'step', 'stepfilled'}
                          align='mid',#箱子边界值的对齐方式，默认为mid{'left', 'mid', 'right'}
                          orientation='vertical',#箱子水平/垂直('vertical'/'horizontal')
                          rwidth=1.0,#箱子宽度，默认为1，此时显示50%
                          log=False,#y轴数据是否取对数
                          color=palettable.colorbrewer.qualitative.Dark2_7.mpl_colors[3],
                          label=feature[0],#图例
                          #normed=0,#功能和density一样，二者不能同时使用
                          facecolor='black',#箱子颜色 
                          edgecolor="black",#箱子边框颜色
                          stacked=False,#多组数据是否堆叠
                          alpha=0.5#箱子透明度
                         )
plt.xticks(bins) #x轴刻度设置为箱子边界

for patch in patches:#每个箱子随机设置颜色
    patch.set_facecolor(random.choice(palettable.colorbrewer.qualitative.Dark2_7.mpl_colors))

#直方图三个返回值
print(n)#频数
print(bins)#箱子边界
print(patches)#箱子数

#直方图绘制分布曲线
plt.plot(bins[:(len(bins)-1)],n,'--',color='#2ca02c')
plt.hist(x=[i+0.1 for i in data_hist],label='new sepal length(cm)',alpha=0.3)
plt.legend()
plt.show()
