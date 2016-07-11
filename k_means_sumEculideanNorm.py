
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


###################################################################################

# the function process_spike_multi will detect spikes from mutiple electrons/cells

# Input: 
# signal_matrix is the signal matrix generated from function multi_electrons_generator
# with dimension num_electron*time

# threshold: is the threshold we set to find the maximum of convulution
# window_height/spike_len is the window parameter we set for convolution


# Output: 
# there are two types of output: 
# final_matrix is signal output for modified k-means clustering
# final_matrix_ecud is for k-means clustering in Euclidean distance
def process_spike_multi(signal_matrix,timeline_matrix,threshold, window_height=2,spike_len=100):
	
	# get number of electron
	shape=signal_matrix.shape
	num_electron=shape[0]
	

	# take convolution of the first row of matrix
	signal1=signal_matrix[0]
	signal_abs=map(abs,signal1)
	
	weights = np.repeat(window_height, spike_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/spike_len

	# get information of spike location
	local_max=detect_peaks(convolution, mph=threshold, mpd=spike_len, show=False)

	# print(local_max-spike_len/2)
	# print('convolution',len(local_max))

	# Initialization
	
	m=len(local_max)
	n=spike_len
	



	# initialize label
	label=[]
	
	num_cell=len(timeline_matrix)


	# initialize empty 3D array for final aligned matrix

	final_matrix=np.zeros((num_electron,m,n))
	final_matrix_ecud=np.zeros((m,1))
	
	# initialize empty matrix for aligned matrix for each electron
	
	detected_spikes=np.zeros((m,n))

	# label each spike in one electron

	label_array=np.zeros(num_cell)
	
	print('local_max',local_max-spike_len/2)
	print('num_cell',num_cell)
	print(timeline_matrix)



	for item in local_max:
		for index in range(num_cell):

			distance=[]
			distance=timeline_matrix[index]+spike_len/2-item

			distance=abs(distance)
			label_array[index]=np.amin(distance)
		#print('label_array',label_array)

		#print('label_array',label_array)
		label.append(np.argmin(label_array))



	# loop over each row of matrix_electron --- every electron
	for num in range(num_electron):
		
		# initialize index
		index=0
		
		signal=signal_matrix[num]

		# locate spike in electron signal and label them
		for item in local_max:
			detected_spikes[index]=signal[item-spike_len/2:item+spike_len/2]
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

		# add the aligned to our final matrix
		final_matrix[num,:,:]=detected_spikes
		final_matrix_ecud=np.concatenate((final_matrix_ecud,detected_spikes),axis=1)
	
		#get rid of the first zeros in the matrix
	n=final_matrix_ecud.shape[1]
	final_matrix_ecud=final_matrix_ecud[:,1:n]

	return final_matrix,final_matrix_ecud,label


#####################################################################################
# k_means_sumEuclidean is the modified k-means clustering for spike detection


def k_means_sumEuclidean(aligned_spikes,num_cluster,iterations=20,spike_len=100):


	#Initialization 


	# here aligned_spikes is 3-D array. Each block(there are num_electron blocks)
	# is a num_spike*spike_dim matrix
	
	num_spike=aligned_spikes.shape[0]
	spike_dim=aligned_spikes.shape[1]
	
	num_electron=int(spike_dim/spike_len)
	spike_dim=spike_dim/num_electron

	#print(aligned_spikes.shape)
	aligned_spikes=aligned_spikes.reshape(num_spike,num_electron,spike_dim)


	# Initialize center of k-means clustering
	k=np.random.permutation(num_spike)
	initial_center=np.zeros((num_electron,num_cluster,spike_dim))

	# return randomly initialized centers
	for num in range(num_cluster):
		initial_center[:,num,:]=aligned_spikes[:,k[num],:]
	
	# Main loop:
	center_vectors=initial_center
	for ite in range(iterations):
		
		# Determine clusters by computing the modified distance(sum of L2 distance of each electron)
		# initialize distance
		electron_distance=np.zeros((num_spike,num_cluster))
		
		for i in range(num_electron):
			spike=aligned_spikes[i,:,:]
			center=center_vectors[i,:,:]
				
			distance_singleEletron=distance.cdist(spike,center,'euclidean')
			
			# Sum over distance in each electron
			electron_distance=electron_distance+distance_singleEletron

		label=electron_distance.argmin(axis=1)


		for index in range(0,num_cluster):
			cluster_vector=aligned_spikes[:,label==index,:]
			number=cluster_vector.shape[1]
			
			# Get new center by averaging	
			center_vectors[:,index,:]=1.0/number*np.sum(cluster_vector,axis=1)			


	center_vector=center_vectors.reshape(1,num_electron,num_cluster,spike_dim).swapaxes(1,2).reshape(num_cluster,-1)

	return center_vector










