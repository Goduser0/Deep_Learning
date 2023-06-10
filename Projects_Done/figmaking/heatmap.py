import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import palettable #python颜色库
from sklearn import datasets 

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

iris=datasets.load_iris()
iris_data,iris_target=iris.data,iris.target

df_iris=pd.DataFrame(np.hstack((iris_data,iris_target.reshape(150,1))),columns=np.append(iris.feature_names,'class'))

data1=np.asarray(df_iris[str(iris.feature_names[0])]).reshape(25,6)
df_data1=pd.DataFrame(data1,[chr(i) for i in range(65,90)],['a','b','c','d','e','f'])
plt.figure(dpi=200)
sns.heatmap(
            data=df_data1,
            # 以下参数均有默认值
            
            # vmin=5,# 设置色条下限
            # vmax=8,# 设置色条上限
            
            # cmap=plt.get_cmap('winter_r'), # 设置colormap（matplotlib内置）
            # cmap=sns.dark_palette("#2ecc71", as_cmap=True),# 设置colormap（seaborn内置暗色色盘）
            # cmap=sns.light_palette("#2ecc71", as_cmap=True),# 设置colormap（seaborn内置亮色色盘）            
            # cmap=sns.diverging_palette(10,220,sep=80,n=7),# 设置colormap（seaborn内置对比度加强色盘）
            cmap=sns.cubehelix_palette(as_cmap=True),# 设置colormap（seaborn内置渐变色盘）
            
            center=7,# colorbar中心数值大小，控制色彩深浅

            annot=True,# 热图中文本显示
            fmt='.2f',# 控制数字格式
            annot_kws={'size':8,'weight':'bold','color':'darkblue'},# 文本格式字体 

            linewidths=1,# 单元格框线宽度
            linecolor='red',# 单元格框线颜色
            
            #colorbar设置
            cbar=True,# 设定colorbar可见性
            cbar_kws={
                'label':'This is the colorbar',
                'orientation':'horizontal', #default:vertical
                'ticks':np.arange(4.5,8,.5),
                'format':'%.3f',
                'pad':0.15
            },
            
            # 设置掩膜（屏蔽原数据）
            mask=df_data1<6.0,

            # 自定义heatmap的x，y标签
            xticklabels=['你','好', '世','界','再','见'],
            yticklabels=True,
           )

plt.title('heatmap')
plt.show()
