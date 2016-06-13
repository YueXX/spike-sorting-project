from __future__ import division, print_function
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal
from sklearn import cluster


import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
#from peak_detect import detect_spike

from k_means import process_spike
from k_means import k_means_spikeDetection
##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells

from data_initilization_2D import spike_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import noise
from data_initilization_2D import plot_spike
#from k_means import detect_spike


time=50000
spike_len=200
lambd=200
shape_parameter=np.array([[0,30],[30,30]])
shape_parameter2=np.array([[0,70],[30,50]])



a=spike_generator(time,spike_len,lambd)
a2=spike_generator(time,spike_len,lambd)


b,c=waveform_generator(a,spike_len,shape_parameter)
b2,c2=waveform_generator(a2,spike_len,shape_parameter2)


# plt.plot(c[0:1000])
# plt.plot(c2[0:1000])


d=c+c2
d=noise(d,0.02)

# plt.plot(d)



#plt.plot(d[0:2000])
# plt.show()
#plt.plot(d[0:1000])
d_new=map(abs,d)


weights = np.repeat(2, 0.6*spike_len)

con=np.convolve(weights,d_new,'same')


con=con/(spike_len)

#a=argrelextrema(con, np.greater)
# plt.plot(con)
# plt.show()

#a=detect_peaks(con, mph=0.4, mpd=100, show=False)

#print(a)


# i=0
# m=len(a)
# n=spike_len/0.2

# detect_spike=np.zeros((m,n))
# for item in a:
# 	detect_spike[i]=d[item-100/2:item+100/2]
# 	# plt.plot(detect_spike)
# 	# plt.show()

# 	i=i+1

# plt.plot(detect_spike[1])
# plt.show()

# plt.plot(d)
# plt.show()
a=process_spike(d,200,0.1)

b=k_means_spikeDetection(a,3,3)


plt.plot(b[0])
plt.plot(b[1])

plt.show()


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
