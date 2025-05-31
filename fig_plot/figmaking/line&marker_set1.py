import matplotlib.pyplot as plt

# marker

# 基本marker
marker_basic=['.', ',', '1', '2', '3', '4', '+', 'x', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

# 高级marker
# https://matplotlib.org/tutorials/text/mathtext.html
# ⨂  \bigotimes
# 使用 marker=r$str$ 使用自定义marker
# 'r$\bigotimes$','$666$'

# 演示
# plt.plot([1,2,3],[1,2,3],marker=r'$\bigotimes$',markersize=20,color='lightblue',label='marker')    
# plt.legend()
# plt.show()



# linestyle

linestyle_str=['-',':','--','-.']
linestyle_tuple=[(0,(1,10)),
                 (0,(1,1)),
                 (0,(1,2)),
                 (0,(5,10)),
                 (0,(5,5)),
                 (0,(5,1)),
                 (0,(3,10,1,10)),
                 (0,(3,5,1,5)),
                 (0,(3,1,1,1)),
                 (0,(3,5,1,5,1,5)),
                 (0,(3,10,1,10,1,10)),
                 (0,(3,1,1,1,1,1))
                 # (0,(line,space,line,space,line,space......))
                ]

# 演示
# for i in linestyle_tuple:
#     plt.plot([1,2,3],[1,2,3],linestyle=i,linewidth=2,color='lightblue',label='linestyle: '+str(i))    
#     plt.legend()
#     plt.show()
