from pickle import TRUE
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文

iris=datasets.load_iris()
iris_data,iris_target=iris.data,iris.target
df_iris=pd.DataFrame(np.hstack((iris_data,iris_target.reshape(150,1))),columns=np.append(iris.feature_names,'class'))
print(iris.feature_names)
print(df_iris[iris.feature_names[1]].describe())

plt.figure(dpi=150)
sns.boxplot(
    # x轴
    x=[iris.target_names[0] if i==0.0 else iris.target_names[1] if i==1.0 else iris.target_names[2] for i in df_iris['class']],
    y=df_iris[iris.feature_names[1]],
    
    orient='v',# 朝向
    
    showfliers=True,# 异常值是否显示
    
    # 离散值marker属性
    fliersize=6,# marker大小    
    flierprops={
        'marker':'o',
        'markerfacecolor':'green',
        'color':'black',
        
    },
    
    # 上下横线属性
    showcaps=True,# 上下横线是否显示
    capprops={'linestyle':'--','color':'red'},
    
    # 上下须线属性
    whiskerprops={'linestyle':'-.','color':'purple'},
    
    notch='True',# 箱子缺口设置
    # 箱子（外框线，填充色）属性
    boxprops={'color':'pink','facecolor':'lightblue'},

    # 中位线设置
    medianprops={'linestyle':'--','color':'red'},

    # 均值点/线显示总开关
    showmeans=True,
    meanprops = {'marker':'D','markerfacecolor':'red'},# 均值点
    # 均值线设置
    meanline=True,# 显示均值线
    # meanprops = {'linestyle':'--','color':'red'},# 设置均值线属性

    # 箱子宽度占比
    width=0.8,
)
plt.show()

# 多类合绘
sns.boxplot(
    x=[iris.target_names[0] if i==0.0 else iris.target_names[1] if i==1.0 else iris.target_names[2] for i in df_iris['class']],
    y=df_iris[iris.feature_names[1]],
    hue=df_iris[iris.feature_names[3]],
    order=["virginica", "versicolor", "setosa"],#设置箱子的显示顺序
    hue_order=sorted(list(df_iris['petal width (cm)'].unique())),#设置每个子类中箱子的显示顺序，此处设置从小到大排序
    orient='v'
)
plt.show()
