from __future__ import division, print_function
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal
from sklearn import cluster
from peak_detect import detect_peaks
from scipy.spatial import distance


import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
#from peak_detect import detect_spike

from k_means import process_spike
from k_means import k_means_spikeDetection
from k_means import plot_kMeans_clusters
##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells

from data_initilization_2D import spike_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import noise
from data_initilization_2D import plot_spike
#from k_means import detect_spike

############################################################
# Generate stimulated data
time=200000
spike_len=500

lambd1=900
lambd2=900
lambd3=900

shape_parameter=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter2=np.array([[0,-50],[40,20],[500,500]])

shape_parameter3=np.array([[0,30],[30,30],[400,700]])





a=spike_generator(time,spike_len,lambd1)
a2=spike_generator(time,spike_len,lambd2)
a3=spike_generator(time,spike_len,lambd3)




b,c=waveform_generator(a,spike_len,shape_parameter)
b2,c2=waveform_generator(a2,spike_len,shape_parameter2)
b3,c3=waveform_generator(a3,spike_len,shape_parameter3)


plt.plot(c[0:2500])
plt.plot(c2[0:2500])
plt.plot(c3[0:2500])


plt.show()

d=c+c2+c3
#d=noise(d,10)

plt.plot(d[0:25000])
plt.show()

##################################################################
# get convolution

window=spike_len*0.5
spike=process_spike(d,window,spike_len/2,40)

for i in range(3):
	plt.plot(spike[i])
plt.show()


b,c=k_means_spikeDetection(spike,3,5)

# a=np.array([[1,2],[3,4]])
# b=np.array([[0,0]])
# c=distance.cdist(a,b,'euclidean',p=2)
# print(c)

plt.plot(b[0])

plt.plot(b[1])

plt.plot(b[2])



plt.show()

plot_kMeans_clusters(3,c)
# plt.plot(b[0])
# plt.plot(b[1])

# plt.show()


# x=np.array([1,2,3,4,4,4,4])
# c=x.argmax(axis=0)
# print(c)
# plt.plot(a[1])
# plt.plot(a[2])

# plt.plot(a[3])
# plt.plot(a[4])



# plt.plot(a[5])
# plt.plot(a[6])


# plt.plot(a[7])
# plt.plot(a[8])


#plt.show()





# data=a
# k = 2
# kmeans = cluster.KMeans(n_clusters=k)
# kmeans.fit(data)

# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

# for i in range(k):
#     # select only data observations with cluster label == i
#     ds = data[np.where(labels==i)]
#     # plot the data observations
#     plt.plot(ds[:,0],ds[:,1],'o')
#     # plot the centroids
#     lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
#     # make the centroid x's bigger
#     plt.setp(lines,ms=15.0)
#     plt.setp(lines,mew=2.0)
# plt.show()


# tel = {'jack': 4098, 'sape': 4139}
#print(a)
#print(con[0:1000])

# plt.plot(con[0:1000])
#plt.show()


# scipy.signal.find_peaks_cwt

# signal.argrelmax(y_array, order=5)
