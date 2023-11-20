from turtle import title
import matplotlib.pyplot as plt

plt.plot([1,2,3],label='label1')
plt.plot([4,7,9],label='label2')

plt.legend(
    # Loc {
    # 'best':0
    # 'upper right':1
    # 'upper left':2 
    # 'lower left':3 
    # 'lower right':4 
    # 'right':5 
    # 'center left':6 
    # 'center right':7 
    # 'lower center':8 
    # 'upper center':9 
    # 'center':10
    loc=6,# 图例位置
    bbox_to_anchor=(0.45,1),# 控制图例相对于figure
    ncol=2,#图例分两行显示
    fontsize=10,# 图例大小
    title='Title Legend',# 图例标题
    title_fontsize=10,# 标题字号
    shadow=True,# 背景阴影
    fancybox=True,# 图例框为圆角
    framealpha=0.6,# 透明度
    facecolor='#BAE4B3',# 背景填充色
    edgecolor='r',# 框线颜色
)

plt.show()