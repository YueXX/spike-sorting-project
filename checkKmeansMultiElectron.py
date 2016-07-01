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

from data_initilization import spike_timeline_generator
from data_initilization import waveform_generator
from data_initilization import noise
from data_initilization import plot_spike
from data_initilization import spike_shape_generator
from data_initilization import multi_electrons_generator
#from k_means import detect_spike

############################################################
# Generate stimulated data
time=100000
spike_len=200

lambd1=1200
lambd2=800


shape_parameter=np.array([[0,20],[30,30],[6000,6000]])
shape_parameter2=np.array([[0,-15],[40,20],[3500,3500]])
shape_parameter3=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter4=np.array([[0,-40],[25,30],[1000,1000]])


sig1=spike_shape_generator(shape_parameter,plot=False)
sig2=spike_shape_generator(shape_parameter2,plot=False)

sig3=spike_shape_generator(shape_parameter3,plot=False)
sig4=spike_shape_generator(shape_parameter4,plot=False)


signa_1=np.concatenate((sig1,sig3), axis=0)
signa_2=np.concatenate((sig2,sig4), axis=0)

plt.plot(signa_1)
plt.plot(signa_2)
plt.savefig('original_signal')


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
noise_level=200
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



# Step 3.1: align spikes 
k=rand.randint(0,m-1)
max_location1=detected_spikes1[k].argmax(axis=0)
max_location2=detected_spikes2[k].argmax(axis=0)

for i in range(0,m-1):
	spike_max_location2=detected_spikes2[i].argmax(axis=0)
	distance2=max_location2-spike_max_location2
	detected_spikes2[i]=np.roll(detected_spikes2[i],distance2)

	spike_max_location1=detected_spikes1[i].argmax(axis=0)
	distance1=max_location1-spike_max_location1
	detected_spikes1[i]=np.roll(detected_spikes1[i],distance1)


# step 4: concatenate the above two matrices

final_matrix=np.concatenate((detected_spikes1, detected_spikes2), axis=1)
print(final_matrix.shape)
# Step 5: apply Kmeans
num_cluster=2
center_vectors,classified_spikes=k_means_spikeDetection(final_matrix,num_cluster,iterations=30)

plt.plot(center_vectors[0])
plt.plot(center_vectors[1])
plt.savefig('Found_center')


plot_kMeans_clusters(num_cluster,classified_spikes)












