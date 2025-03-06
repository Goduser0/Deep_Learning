# 散点图

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

iris=datasets.load_iris()
# 数据集属性['DESCR','data','feature_names','target','target_name']
x,y=iris.data,iris.target
pd_iris=pd.DataFrame(np.hstack((x,y.reshape(150,1))),columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','class'])

plt.figure(dpi=150)
iris_type=pd_iris['class'].unique() #重点
i_name=iris.target_names #重点
i_marker=['.',',','$\clubsuit$']
i_color=['#c72e29','#098154','#fb832d']

for i in range(len(iris_type)):
    plt.scatter(
                pd_iris.loc[pd_iris['class']==iris_type[i],'sepal length (cm)'],#重点
                pd_iris.loc[pd_iris['class']==iris_type[i],'sepal width (cm)'],#重点
                s=50,
                color=i_color[i],
                alpha=0.8,
                # facecolor='r',
                # edgecolors='none',
                marker=i_marker[i],
                linewidths=1,
                label=i_name[i]
                )
plt.legend()
plt.show()


