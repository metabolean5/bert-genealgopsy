# 1. Import Modules
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use("seaborn")
 
# 2. Generate a 10x10 random integer matrix


data = np.array([[29,  6, 17],
                [ 3, 67, 13],
                [ 2, 13, 51]])


print("Our dataset is : ",data)
 
# 3. Plot the heatmap
plt.figure(figsize=(3,3))
heat_map = sns.heatmap( data, linewidth = 1 , annot = True, cmap="Blues")
heat_map.set_yticklabels(['1-2','3-4','5'])
heat_map.set_xticklabels(['1-2','3-4','5'])
plt.title( "HeatMap using Seaborn Method" )
plt.show()


'''svm original
Confusion matrix:
[[20  0  9  1 13]
 [ 1  0  0  0  0]
 [ 1  0 38  5 21]
 [ 4  0  8  4  6]
 [ 4  0 23  5 38]]
 '''

'''svm original no 5
Confusion matrix:
[[27  0 15  1]
 [ 0  0  3  0]
 [ 3  0 57  3]
 [ 2  0 19  6]]
 '''

'''svm merge original
Confusion matrix:
[[13 16  9]
 [ 0 85  9]
 [ 1 35 33]]
 '''
