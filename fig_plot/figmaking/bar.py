from turtle import color
import matplotlib.pyplot as plt
import numpy as np

# bar:垂直，水平，堆积

# 垂直
plt.figure(dpi=150)
labels=['Jack','Rode','Jimmy']
year_2019=np.arange(1,4)
plt.bar(
        np.arange(3,6),# 设定各条形中心的x位置
        year_2019,# 设定各条形高度
        width=0.4,# 设定各条形宽度
        bottom=0,# 设定各条形底部初始位置（y坐标）
        align='center',# 设定各条形名称与条形的相对位置
        color='pink',# 设定各条形填充色
        edgecolor='b',#　设定外框色
        linewidth=1,# 设定外框线宽度
        tick_label=labels,# 自定义各条形名称
        yerr=np.arange(0.1,0.35,0.1),# 添加误差棒
        ecolor='red',# 误差棒颜色
        capsize=5,# 误差棒上下横线长度
        log=False# y坐标是否取对数
)
plt.show()

# 多条形垂直（相当于绘制多次单条形图，错开x位置）
year_2020=np.arange(1,4)+1
bar_width=0.4
bar1=plt.bar(np.arange(len(labels))-bar_width/2,#为了两个柱子一样宽
        year_2019,
        color='#B5495B',
        width=bar_width, 
        label='year_2019'#图例
        
       )
bar2=plt.bar(np.arange(len(labels))+bar_width/2,
        year_2020,
        color='#2ca02c',
        width=bar_width,
        label='year_2020'#图例
        
       )
plt.xticks(np.arange(0, 3, step=1),labels,rotation=45)#定义柱子名称
plt.legend(loc=2)#图例在左边

# 在条形上添加文字
def autolabel(rects):
    """柱子上添加文字"""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(
                     '{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height+0.1),# 设定文字位置基准
                     xytext=(0, 10),# 设定文字相对于位置基准的坐标
                     textcoords="offset points",
                     ha='center', va='bottom'
                    )
autolabel(bar1)
autolabel(bar2)
plt.show()

# 堆积条形图（相当于在一个bar上添加一个bar，后者起始点为前者的高度）
plt.bar(
        np.arange(len(labels)),
        year_2019,
        color='r',
        label='2019',
        width=0.4
)
plt.bar(
        np.arange(len(labels)),
        year_2020,
        color='b',
        label='2020',
        bottom=year_2019,
        width=0.3
)
plt.legend()
plt.show()

# 水平柱状条形图
plt.barh(np.arange(len(labels)),#每个柱子的名称
        width=year_2019,#柱子高度
        height=0.8,#柱子宽度，默认为0.8
        left=1,#柱子底部位置对应x轴的横坐标，类似bar()中的bottom         
        align='center',#柱子名称位置，默认为'center'，可选'edge'
        color='pink',
        edgecolor='b',
        linewidth=1,
        tick_label=labels,#自定义柱子名称
        xerr=[0.1,0.2,0.3],#添加误差棒
        ecolor='red',#误差棒颜色
        capsize=5,
        log=False,#x轴坐标是否取对数       
        )
plt.show()