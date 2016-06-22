# simple function to test k-means on 2 cells/2 electrons 


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

from data_initilization_2D import spike_timeline_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import noise
from data_initilization_2D import plot_spike
#from k_means import detect_spike

############################################################
# Generate stimulated data
time=80000
spike_len=500

lambd1=1200
lambd2=800


shape_parameter=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter2=np.array([[0,-50],[40,20],[1000,1000]])
shape_parameter3=np.array([[0,30],[30,30],[1000,800]])
shape_parameter4=np.array([[0,-20],[25,30],[2000,1500]])

a=spike_timeline_generator(time,lambd1,False,spike_len)
a2=spike_timeline_generator(time,lambd2,False,spike_len)

c=waveform_generator(a,shape_parameter,spike_len)
c2=waveform_generator(a2,shape_parameter2,spike_len)

c3=waveform_generator(a,shape_parameter3,spike_len)
c4=waveform_generator(a2,shape_parameter4,spike_len)


signal1=c+c2
signal1=noise(signal1,5)

signal2=c3+c4
signal2=noise(signal2,5)

# plt.plot(signal1[0:9000])
# plt.show()

# plt.plot(signal2[0:9000])
# plt.show()



# how to apply k-means to different electrons:
# 1. use multi-electrons generator to generate a signal matrix of 2 electrons and 
# multiple cells
# 2. apply convolution on the first row of the signal matrix to get the spike location
# 3. locate spikes on the other rows using the same information from step 2
# 4. concate the above result matrix
# 5. apply k-means to the final matrix


# step 2:

window_len=spike_len/2
take_window_len=spike_len/2

signal_abs=map(abs,signal1)
	
# Step 2: take convolution of the absolute value
window_height=2
weights = np.repeat(window_height, window_len)
convolution=np.convolve(weights,signal_abs,'same')
convolution=convolution/window_len

# plt.plot(convolution[0:2000])
# plt.show()


# get the location of spikes in signal 1
noise_level=100
local_max=detect_peaks(convolution, mph=noise_level*5, mpd=window_len, show=True)


# Step 3: get the spikes in signal1 and signal2 using information from step 2 
m=len(local_max)
n=take_window_len
detected_spikes1=np.zeros((m,n))
detected_spikes2=np.zeros((m,n))

index=0
for item in local_max:
	detected_spikes1[index]=signal1[item-take_window_len/2:item+take_window_len/2]
	detected_spikes2[index]=signal2[item-take_window_len/2:item+take_window_len/2]
	index=index+1



print(detected_spikes1.shape)
print(detected_spikes2.shape)

# step 4: concatenate the above two matrices

final_matrix=np.concatenate((detected_spikes1, detected_spikes2), axis=1)

# Step 5: apply Kmeans
num_cluster=2
center_vectors,classified_spikes=k_means_spikeDetection(final_matrix,num_cluster,iterations=10)
plot_kMeans_clusters(num_cluster,classified_spikes)












