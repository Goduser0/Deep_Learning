from turtle import color
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib import cm
import numpy as np

# 颜色表示
# RGB/RGBA 元组格式: (0.1,0.2,0.5)/(0.1,0.2,0.5,0.3)
# RGB/RGBA hex格式: '#0F0F0F'/'#0F0F0F0F'
# 0~1 float: '0.6'
# 基本色: 'b' 'g' 'r' 'c' 'm' 'y' 'k' 'w'

# 内置颜色
# matplotlib.colors.BASE_COLORS
# matplotlib.colors.TABLEAU_COLORS
# matplotlib.colors.CSS4_COLORS
# matplotlib.colors.XKCD_COLORS
# print(mpcolors.CSS4_COLORS)

# 内置颜色条
# print(dir(cm))

# cm_name:色条的名称 int:整数
# 取离散色条多色
# color=plt.cm.get_camp('cm_name')(range(start,end))
# 取离散色条单色
# color=plt.cm.get_camp('cm_name')(float)
# 取连续色条多色
# color=plt.get_cmap('cm_name')(linespace(start,end,num_part))
# 取连续色条单色
# color=plt.get_camp('cm_name')(float)

plt.subplot(1,1,1)
plt.bar(range(5),range(1,6),color=plt.cm.get_cmap('hsv')(0.2))
plt.show()
