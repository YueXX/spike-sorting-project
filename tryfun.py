import numpy as np
import random as rand
from scipy.spatial import distance


# x=np.array([[1,2,3,7],[2,3,4,7],[3,4,5,7]])

# y=x.reshape(3,2,2)

#print(y[:,1,:])

shape=[]
x=np.array([1,2,3])

centers=np.array([[-10,0,3],[3,1,4],[0,4,-40]])

# a=distance.cdist(x,centers,'euclidean')

# c=np.where(centers == np.min(centers))
# index=(np.unravel_index(centers.argmin(),centers.shape))
# index=np.array(index)

print(x)