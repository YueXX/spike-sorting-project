
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

from k_means import process_spike
from k_means import k_means_spikeDetection
from k_means import plot_kMeans_clusters

from data_initilization import spike_timeline_generator
from data_initilization import waveform_generator
from data_initilization import noise
from data_initilization import multi_electrons_generator
################################################################################################

# how to apply k-means to different electrons:
# 1. use multi-electrons generator to generate a signal matrix of multiple electrons and 
# multiple cells(aligned signals)
# 2. apply convolution on the first row(or any row, since aligned)of the signal matrix to get the spike location
# 3. locate spikes on the other rows using the spike location information from step 2
# 4. concate the above result matrix
# 5. apply k-means to the final matrix

# Step1:
# Generate data (mutiple electrons, aligned) to apply k means

# our signal matrix will be matrix_electron

# step 2: apply convolution on the first row of the signal matrix to get the spike location

# step2.1: convolution
def process_spike_multi(signal_matrix,threshold, window_height=2,window_len=100):
	
	# get number of electron
	shape=signal_matrix.shape
	num_electron=shape[0]
	
	# take convolution of the first row of matrix
	signal1=matrix_electron[0]
	signal_abs=map(abs,signal1)
	
	weights = np.repeat(window_height, window_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/window_len

	# get information of spike location
	local_max=detect_peaks(convolution, mph=threshold, mpd=window_len, show=True)


	# initialize empty matrix for final aligned matrix
	m=len(local_max)
	n=window_len

	final_matrix=np.zeros((m,n))

	# initialize empty matrix for aligned matrix for each electron
	
	detected_spikes=np.zeros((m,n))


	# loop over each row of matrix_electron --- every electron
	for num in range(num_electron):
		
		# initialize index
		index=0
		
		signal=matrix_electron[num]

		# locate spike in electron signal
		for item in local_max:
			detected_spikes[index]=signal[item-window_len/2:item+window_len/2]
			index=index+1
		# aligned detected spikes in one signel electron

		# random choose one row 
		k=rand.randint(0,m-1)
		# get the maximum location of this row
		max_location=detected_spikes[k].argmax(axis=0)

		# roll other rows according to this information
		for i in range(0,m):
			spike_max_location=detected_spikes[i].argmax(axis=0)
			distance=max_location-spike_max_location
			detected_spikes[i]=np.roll(detected_spikes[i],distance)

		print(detected_spikes.shape)
		# add the aligned to our final matrix
		final_matrix=np.concatenate((final_matrix,detected_spikes),axis=1)


	return final_matrix



########################################################
#test data
num_electron=2
num_cell=2
time=30000
noise_level=0.01
overlap_level=1000
boolean=False
plot=True
threshold=100

matrix_electron, boolean, spike_shape_parameter=multi_electrons_generator(num_electron,num_cell,time)



#process data

a=process_spike_multi(matrix_electron,threshold)

num_cluster=2
interations=20

	# apply k-means
center_vectors, classified_spikes=k_means_spikeDetection(a,num_cluster,interations)

	# plot result
plot_kMeans_clusters(classified_spikes,center_vectors,num_cluster)

print('Ahhhh')





