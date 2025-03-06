from asyncio import as_completed
from cmath import pi
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

x_plot=np.linspace(0,4*pi,200)
y_plot=np.array([math.sin(x) for x in x_plot])

plt.figure(dpi=150)
plt.plot(
        x_plot,
        y_plot,
        linestyle=(0,(3,1,2,1,1)),
        linewidth=1,
        color='#3f9c5a',

        marker='*',
        markersize=12,
        markeredgewidth=1,
        markerfacecolor='r',# 填充部分颜色
        fillstyle='top',# 填充方式
        markerfacecoloralt='b',# 未填充部分颜色
        markevery=10,
        label='sin(x)'
        )
plt.legend()
plt.show()