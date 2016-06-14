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
time=800000
spike_len=500

lambd1=1200
lambd2=1200
lambd3=1200

shape_parameter=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter2=np.array([[0,-50],[40,20],[1000,1000]])
shape_parameter3=np.array([[0,30],[30,30],[400,300]])


a=spike_generator(time,spike_len,lambd1)
a2=spike_generator(time,spike_len,lambd2)
a3=spike_generator(time,spike_len,lambd3)


b,c=waveform_generator(a,spike_len,shape_parameter)
b2,c2=waveform_generator(a2,spike_len,shape_parameter2)
b3,c3=waveform_generator(a3,spike_len,shape_parameter3)

# Plot data sample 
plt.plot(c[0:2500])
plt.plot(c2[0:2500])
plt.plot(c3[0:2500])
plt.show()

# generate a single electron
d=c+c2+c3
# add noise
d=noise(d,10)


plt.plot(d[0:25000])
plt.show()

##################################################################
# apply k-means algorithm

# process spike
window_len=spike_len/2
take_window_len=spike_len/2

noise_level=40
aligned_spike=process_spike(d,window_len,take_window_len,noise_level)

# for i in range(3):
# 	plt.plot(spike[i])
# plt.show()

# apply k_means

num_cluster=3
interations=10
center_vectors, classified_spikes=k_means_spikeDetection(aligned_spike,num_cluster,interations)


plot_kMeans_clusters(num_cluster,classified_spikes)


















