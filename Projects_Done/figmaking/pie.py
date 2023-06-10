import matplotlib.pyplot as plt

my_dpi=96
plt.figure(figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi)

plt.pie(
    x=[0.36,0.42,0.62],
    labels=['A','B','C'],
    colors=['#d5695d','#5d8ca8','#65a479'],#传入颜色list
    explode=(0,0.1,0),# 控制爆炸突出显示
    autopct='%.2f%%',# 格式化输出百分比
    pctdistance=0.8,# 控制百分比标签与圆心的距离
    labeldistance=1.5,# 控制类别标签与圆心的距离
    startangle=45,# 开始绘制角度
    radius=1.5,# pie的半径
    counterclock=False,# 控制顺/逆时针展示
    wedgeprops={# 框线设置
        'edgecolor':'r',
        'linestyle':'-.',
        'alpha':0.5,
    },
    textprops={# 设置文本属性
        'color':'r',
        'fontsize':16,
        'fontfamily':'Microsoft JhengHei'
    },

)

plt.legend()
plt.show()
