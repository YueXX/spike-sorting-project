
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

from k_means_Eculidean2Norm import process_spike
from k_means_Eculidean2Norm import k_means_spikeDetection
from k_means_Eculidean2Norm import plot_kMeans_clusters

from data_initilization import spike_timeline_generator
from data_initilization import waveform_generator
from data_initilization import noise
from data_initilization import multi_electrons_generator
from k_means_sumEculideanNorm import process_spike_multi
from k_means_sumEculideanNorm import k_means_sumEuclidean
from k_means_sumEculideanNorm import classify_label
from k_means_sumEculideanNorm import prediction_rate
from SumEculideanKmeans_spikeDetection import Kmeans_spikeDetection

# Main file
# test functions 

#test data
def main():

	num_electron=3
	num_cell=3
	time=40000
	delay=False
	noise_level=0.01
	overlap_level=1000
	boolean=False
	plot=False
	threshold=80

	matrix_electron, boolean,signal_cell_electron,spike_timeline_parameter,spike_shape_parameter=multi_electrons_generator(num_electron,num_cell,time,delay,noise_level,overlap_level,boolean,plot,spike_len=100)


	#print(spike_timeline_parameter[0])

	#print(spike_shape_parameter)


	final_matrix,final_matrix_ecud,label=process_spike_multi(matrix_electron,spike_timeline_parameter,threshold)
	
	classify_label(final_matrix,label,"real")
	num_cluster=num_cell

	#center_vectors,final_label=k_means_sumEuclidean(final_matrix,num_cluster)

	kMeans=Kmeans_spikeDetection(num_cluster=3,iterations=10,distance_mode='sum')
	
	#kMeans=Kmeans_spikeDetection(num_cluster=3,iterations=10,distance_mode='SumEculidean')
	
	kMeans.fit(final_matrix)
	kMeans.plotCluster(final_matrix)



	# #print('Label sumEculidean')
	# classify_label(final_matrix,final_label,"sumEculidean")

	# center_vectors2,final_label2=k_means_spikeDetection(final_matrix_ecud,num_cluster)

	# #print('Label Eculidean')
	
	# classify_label(final_matrix,final_label2,"Eculidean")

	# print('error rate sumEculidean')
	
	# prediction_rate(label,final_label)

	# print('error rate Eculidean')
	# prediction_rate(label,final_label2)
	# #plt.plot(x[0])
	
	#plt.plot(x2[0])



	#plt.show()

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








