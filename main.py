
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
from k_means_multi import process_spike_multi
from k_means_multi import k_means_sumEuclidean


# Main file
# test functions 

#test data
def main():

	num_electron=2
	num_cell=2
	time=30000
	noise_level=0.01
	overlap_level=500
	boolean=False
	plot=True
	threshold=100

	matrix_electron, boolean, spike_shape_parameter=multi_electrons_generator(num_electron,num_cell,time)
		
	a1,a2=process_spike_multi(matrix_electron,threshold)

	x=k_means_sumEuclidean(a1,2,50)

	x2,y2=k_means_spikeDetection(a2,2,50)

	plt.plot(x[1])
	plt.plot(x2[1])
	plt.show()

	# x = np.arange(16).reshape((4,2,2))
	# x=x.reshape(1,4,2,2).swapaxes(1,2).reshape(2,-1)
	
	# print(x)
	# #print(a[:,0,:].shape)
	# plt.plot(a[1,:])
	# # plt.plot(a[:,0,:])

	# plt.show()


	# x=np.array([[1,2],[3,4],[5,6]])
	# y=np.argmin(x,axis=1)
	# #y=y.reshape(1,3)
	# print(y)



	# y=np.array([[0,0],[0,0]])

	# xy=distance.cdist(x,y,'euclidean')

	# print(xy)
	# num_cluster=2
	# interations=20

	# 		# apply k-means
	# center_vectors, classified_spikes=k_means_spikeDetection(a,num_cluster,interations)

	# 		# plot result
	# plot_kMeans_clusters(classified_spikes,center_vectors,num_cluster)




if __name__ == "__main__":
	main()








