
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
	num_cell=2
	num_electron=2
	
	overlap_level1=300
	overlap_level2=100

	noise_level=20
	data=data_initialization_spikeSorting(num_cell=num_cell,num_electron=num_electron,time=3000,delay=False)
	
	# shape of the spike is fixed here
	data.data_init()
	#boolean=np.ones((num_cell,num_electron))
	data.signal_generator(overlap_level=overlap_level1,noise_level=noise_level,detect_range=0.2)
	print(data.boolean)

	print('boolean')

	data.plot()
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

	print(modified_signal_matrix1.shape)

	# print(data.true_label,'true_label')
	# print(data.boolean)

# # ##############################################################
# # # # Apply k-means

	kMeans=Kmeans_spikeDetection(num_cluster=num_cell,num_electron=num_electron,iterations=1)
 # 	#print(data.true_label)
	mode='MinEculidean'
	# kMeans.fit(modified_signal_matrix1,distance_mode=mode)
	# kMeans.evaluate(true_label1,kMeans.predict_label_list,mode)
	
	# kMeans.fit(modified_signal_matrix2,distance_mode=mode)
	# kMeans.evaluate(true_label2,kMeans.predict_label_list,mode)

	kMeans.min_threshold(modified_signal_matrix1,distance_mode=mode)
	kMeans.evaluate(true_label1,kMeans.predict_label_list,mode)
	
	# kMeans.min_threshold(modified_signal_matrix2,distance_mode=mode)
	# kMeans.evaluate(true_label2,kMeans.predict_label_list,mode)



	# os.system('say "your code has finished"')



if __name__ == "__main__":
	main()







