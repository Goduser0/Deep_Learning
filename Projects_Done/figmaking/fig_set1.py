import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator

# 载入中文字体
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 设定画布大小
plt.figure(dpi=150)

# 设定总标题
plt.suptitle('总标题',
             x=0.5,# x轴位置
             y=0.99,# y轴位置
             size=15,# 字体大小
             ha='center',# 水平对齐方式
             va='top',# 垂直对齐方式
             weight='bold',# 字体粗细
             rotation=0,# 旋转角度
             )
# 设定子图数量
plt.subplot(1,1,1)

# 设定自定义字体 fontdict关键字选用
font_self1={'family':'Microsoft YaHei','fontsize':12,'fontweight':'bold','color':(.01,.99,.99)}

# 子图标题
plt.title('fig1 title')

# 子图框线显示
plt.gca().spines['top'].set_visible(True) # top/bottom/left/right
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_linestyle('--')

# 网格线显示
plt.grid(True)# axis='x'

#坐标轴刻度（tick）与刻度值（tick label）操作
plt.tick_params(
                #设置操作对象
                axis='both',#对那个方向（x方向：上下轴；y方向：左右轴）的坐标轴上的tick操作，可选参数{'x', 'y', 'both'}
                which='both',#对主刻度还是次要刻度操作，可选参数为{'major', 'minor', 'both'}
                
                #以下四个参数控制上下左右四个轴的刻度的关闭和开启
                top='on',
                bottom='on',
                left='off',
                right='off',
                
                #刻度线设置
                colors='r',
                direction='out',#方向，可选参数{'in', 'out', 'inout'}                
                length=10,
                width=1,
                pad=2,#与刻度值之间的距离
                zorder=0,
                
                #刻度值显示
                labelsize=10,
                labelcolor='#0011ff',
                labeltop='off',
                labelbottom='off',                
                labelleft='off',
                labelright='off',
                labelrotation=30,
                
                # 网格线设置
                grid_color='pink',
                grid_alpha=1,#透明度
                grid_linewidth=1,
                grid_linestyle='-'
               )
# 设置x，y轴范围
plt.xlim(0,2)
plt.ylim(0,2)

# 设置大刻度与小刻度（）
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))#大刻度在自定义标签后失效
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.02))

# 自定义刻度值标签
plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghig'),rotation=45)
plt.yticks([])#关闭刻度值

# 设置x，y轴名称
plt.xlabel('fig1 xlabel')
plt.ylabel('fig1 ylabel')

plt.show()
