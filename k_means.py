import numpy as np
import random as rand
from matplotlib import pyplot as plt
from peak_detect import detect_peaks
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial import distance
#########################################################
# The following code apply K-means algorithms to signal
# by three steps: spike detection, alignment of spikes and k-means algorithm


# Question 1: we have to manually set the window length/convolution window,  
# since we use convolution window to find spike, the convolution result may not give
# us the best estimate

# Question 2: align spikes: better way to locate the spikes instead of align the maximum? ..

#########################################################
# process_spike function will detect spikes in input signal and align them :
# input: 
# signal: the input signal
# window_len: the manually set length for window in convolution
# window_height: the manually set height for window in convolution
# noise_level: the lower bound parameter in the find local maxima function

# output: the aligned spikes in a 2-D array detected_spikes


def process_spike(signal, window_len, noise_level, window_height=2):
	
	################################################
	# Step 1: take the absolute value of signal
	signal_abs=map(abs,signal)

	# Step 2: take convolution of the absolute value
	weights = np.repeat(window_height, window_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/window_len

	# Step 3: find the indices of local maxima of the convolution
	local_max=detect_peaks(convolution, mph=noise_level*5, mpd=window_len/2, show=False)

	# Step 4: locate/save spike vectors
	m=len(local_max)
	n=window_len
	detected_spikes=np.zeros((m,n))
	index=0
	for item in local_max:
		detected_spikes[index]=signal[item-window_len/2:item+window_len/2]
		index=index+1

	# Step 5: align spikes 
	k=rand.randint(0,m-1)
	max_location=detected_spikes[k].argmax(axis=0)
	for i in range(0,m-1):
		spike_max_location=detected_spikes[i].argmax(axis=0)
		distance=max_location-spike_max_location
		detected_spikes[i]=np.roll(detected_spikes[i],distance)

	return detected_spikes


def k_means_spikeDetection(aligned_spikes,num_cluster,iterations=50):
	
	# Initialize spikes with lables
	m=aligned_spikes.shape[0]
	n=aligned_spikes.shape[1]
	

	# Take initialize centers
	k=np.random.permutation(m)
	initial_center=np.zeros((num_cluster,n))
	center_vectors_label=np.zeros((num_cluster,n+1))
	
	for num in range(num_cluster):
		initial_center[num]=aligned_spikes[k[num]]

	# Main loop:
	center_vectors=initial_center
	for ite in range(1,iterations):
		# Determine clusters by computing the Eculidean distance
		clusters_distance=distance.cdist(aligned_spikes,center_vectors,'euclidean',p=2)
		label=clusters_distance.argmin(axis=1)
		classified_spikes=np.c_[aligned_spikes,label]


		for index in range(0,num_cluster):

			cluster_vector=classified_spikes[classified_spikes[:,-1]==index]
			
			number=cluster_vector.shape[0]

			center_vectors_label[index]=1/number*np.sum(cluster_vector,axis=0)
			
			center_vectors=np.delete(center_vectors_label,-1,1)

			#print(cluster_vector)


	return center_vectors






























