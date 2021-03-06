
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
import matplotlib as mpl
from matplotlib.pyplot import cm 


#########################################################
# The following code apply K-means algorithms to signal
# by three steps: spike detection, alignment of spikes and k-means algorithm


# Question 1: we have to manually set the window length/convolution window,  
# since we use convolution window to find spike, the convolution result may not give
# us the best estimate

# Question 2: local maximum determines our result in k means

###########################################################################

# process_spike function will detect spikes in input signal and align them :
# input: 
# signal: the input signal
# window_len: the manually set length for window in convolution
# take_window_len: chop off length for spike
# window_height: the manually set height for window in convolution
# noise_level: the lower bound parameter in the find local maxima function

# output: the aligned spikes in a 2-D array detected_spikes

def process_spike(signal, window_len, take_window_len,noise_level, window_height=2):
	
	################################################
	# Step 1: take the absolute value of signal
	
	signal_abs=map(abs,signal)
	#signal_abs=np.array(signal)**2
	
	# Step 2: take convolution of the absolute value
	weights = np.repeat(window_height, window_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/window_len

	plt.plot(convolution)
	plt.show()

	# Step 3: find the indices of local maxima of the convolution
	local_max=detect_peaks(convolution, mph=noise_level*5, mpd=window_len,threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False)

	# Step 4: locate/save spike vectors
	m=len(local_max)
	n=take_window_len
	detected_spikes=np.zeros((m,n))
	index=0
	for item in local_max:
		detected_spikes[index]=signal[item-take_window_len/2:item+take_window_len/2]
		index=index+1
	detected_spikes1=detected_spikes.copy()

	#return detected_spikes

	# Step 5: align spikes 
	k=rand.randint(0,m-1)
	max_location=detected_spikes[k].argmax(axis=0)
	for i in range(0,m-1):
		spike_max_location=detected_spikes[i].argmax(axis=0)
		distance=max_location-spike_max_location
		detected_spikes[i]=np.roll(detected_spikes[i],distance)

	return detected_spikes



#######################################################################
# K_means_spikeDetection function will perform k-means algorithm on 
# aligned spikes


def k_means_spikeDetection(aligned_spikes,num_cluster,iterations=20):
	# Initialize spikes with lables
	m=aligned_spikes.shape[0]#num of points
	n=aligned_spikes.shape[1]#dim of the points
	
	# Take initialize centers
	k=np.random.permutation(m)
	initial_center=np.zeros((num_cluster,n))

	#return initial_center
	for num in range(num_cluster):
		initial_center[num]=aligned_spikes[k[num]]

	# Main loop:
	center_vectors=initial_center
	for ite in range(iterations):
		
		# Determine clusters by computing the Eculidean distance
		clusters_distance=distance.cdist(aligned_spikes,center_vectors,'euclidean',p=2)
		label=clusters_distance.argmin(axis=1)
		
		classified_spikes=np.c_[aligned_spikes,label]

		# assign each vector in aligned_spikes a group		
		for index in range(0,num_cluster):
			cluster_vector=aligned_spikes[label==index]
			number=cluster_vector.shape[0]

			# Get new center by averaging vectors in a certain group
			center_vectors[index]=1.0/number*np.sum(cluster_vector,axis=0)			

	return center_vectors,label


def plot_kMeans_clusters(classified_spikes,center_vectors,num_cluster):
	color=cm.rainbow(np.linspace(0,1,num_cluster))

	for i in range(num_cluster):
		plt.plot(center_vectors[i])
		plt.savefig('image/classifed_centers.png')

	for index_i in range(0,num_cluster):
		cluster_vector=classified_spikes[classified_spikes[:,-1]==index_i]
		number=cluster_vector.shape[0]

		for index_j in range(0,number):
			#plt.subplot(index)
			plt.plot(cluster_vector[index_j],color=color[index_i])
			#plt.show()

		plt.savefig('image/clusters.png')
























