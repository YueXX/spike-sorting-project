
import numpy as np
from Kmeans_spikeDetection import Kmeans_spikeDetection
from data_initialization_spikeSorting import data_initialization_spikeSorting
from sklearn import cluster, datasets
from matplotlib import pyplot as plt
import sys
import os

def main():
	sys.dont_write_bytecode = True





	# set parameter
	num_cell=6
	num_electron=6
	
	overlap_level1=1000
	overlap_level2=100

	noise_level=20
	data=data_initialization_spikeSorting(num_cell=num_cell,num_electron=num_electron,time=70000,delay=False)
	
	# shape of the spike is fixed here
	data.data_init()
	
	#boolean=np.ones((num_cell,num_electron))
	data.signal_generator(overlap_level=overlap_level1,noise_level=noise_level,detect_range=0.5)
	#data.plot()
	data.process_spike()

	modified_signal_matrix1=data.modified_signal_matrix
	signal_matrix1=data.signal_matrix
	true_label1=data.true_label



	data.signal_generator(overlap_level=overlap_level2,noise_level=noise_level,detect_range=0.5)
	#data.plot()
	data.process_spike()
	signal_matrix2=data.signal_matrix


	modified_signal_matrix2=data.modified_signal_matrix
	true_label2=data.true_label



	# print(data.true_label,'true_label')
	# print(data.boolean)

# # ##############################################################
# # # # Apply k-means

	kMeans=Kmeans_spikeDetection(num_cluster=num_cell,num_electron=num_electron,iterations=40)
 # 	#print(data.true_label)
	mode='MinEculidean'
	kMeans.min_threshold(modified_signal_matrix1,distance_mode=mode)
	kMeans.evaluate(true_label1,kMeans.predict_label_list,mode)
	
	kMeans.min_threshold(modified_signal_matrix2,distance_mode=mode)
	kMeans.evaluate(true_label2,kMeans.predict_label_list,mode)


	#kMeans.plotCenter()

	#print(kMeans.predict_label,'predict_label')

	#kMeans.plotCluster(data.modified_signal_matrix,kMeans.predict_label)

	mode='Eculidean'
	kMeans.fit(signal_matrix1,distance_mode=mode)
	kMeans.evaluate(true_label1,kMeans.predict_label_list,mode)
	
	kMeans.fit(signal_matrix2,distance_mode=mode)
	kMeans.evaluate(true_label2,kMeans.predict_label_list,mode)
	
	mode='SumEculidean'

	kMeans.fit(modified_signal_matrix1,distance_mode=mode)
	kMeans.evaluate(true_label1,kMeans.predict_label_list,mode)
	
	kMeans.fit(modified_signal_matrix2,distance_mode=mode)
	kMeans.evaluate(true_label2,kMeans.predict_label_list,mode)
	
	os.system('say "your code has finished"')



	#kMeans.plotCluster(data.modified_signal_matrix,kMeans.predict_label)

	# kMeans.fit(data.modified_signal_matrix,distance_mode='SumEculidean')
	# kMeans.evaluate(data.true_label,kMeans.predict_label_list)
  # 	kMeans.plotCenter()
	#kMeans.plotCluster(data.modified_signal_matrix,kMeans.predict_label)

	#kMeans.plotCluster(data.modified_signal_matrix,kMeans.predict_label)


	# kMeans.fit(data.modified_signal_matrix,distance_mode='MinEculidean')
	# kMeans.plotCluster(data.modified_signal_matrix,kMeans.label)
	# kMeans.fit(data.signal_matrix,distance_mode='SumEculidean')
	
	# kMeans.plotCluster(data.modified_signal_matrix,kMeans.label)

	# k_means = cluster.KMeans(n_clusters=num_cell)
	# k_means.fit(data.signal_matrix)
	# lis=[k_means.labels_]
	# kMeans.evaluate(data.true_label,lis)
	# kMeans.plotCluster(data.modified_signal_matrix,k_means.labels_)
# 	kMeans.distance_mode='sklearn'
# 	kMeans.plotCluster(data.aligned_matrix_3D,k_means.labels_)
	


if __name__ == "__main__":
	main()








