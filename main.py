
import numpy as np
from Kmeans_spikeDetection import Kmeans_spikeDetection
from data_initialization_spikeSorting import data_initialization_spikeSorting

def main():

	data_1=data_initialization_spikeSorting(num_cell=2,num_electron=3,time=30000,delay=False)

	data_1.data_init()
	boole=np.ones((2,3))
	
	data_1.signal_generator(overlap_level=200,noise_level=5,boolean=boole)
	
	data_1.process_aligned_signal()


	kMeans=Kmeans_spikeDetection(num_cluster=3,iterations=20,distance_mode='SumEculidean')
	
	kMeans.fit(data_1.aligned_matrix_3D)
	kMeans.plotCluster(data_1.aligned_matrix_3D)



if __name__ == "__main__":
	main()








