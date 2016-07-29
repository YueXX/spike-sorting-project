
import numpy as np
from Kmeans_spikeDetection import Kmeans_spikeDetection
from data_initialization_spikeSorting import data_initialization_spikeSorting
from sklearn import cluster, datasets
from matplotlib import pyplot as plt

def main():

# initialize data
	num_cell=4
	num_electron=4
	overlap_level=300
	noise_level=5

	data=data_initialization_spikeSorting(num_cell=num_cell,num_electron=num_electron,time=30000,delay=False)
	data.data_init()
	
	boolean=np.array([[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,0,0,1]])

	boolean=np.ones((num_cell,num_electron))


	data.signal_generator(overlap_level=overlap_level,noise_level=noise_level,boolean=boolean)
	data.plot()
	data.get_aligned_signal()

##############################################################
# Apply k-means

	kMeans=Kmeans_spikeDetection(num_cluster=num_cell,iterations=50)

	kMeans.fit(data.aligned_matrix_3D,distance_mode='SumEculidean')
	kMeans.plotCluster(data.aligned_matrix_3D,kMeans.label)

	kMeans.fit(data.aligned_matrix_3D,distance_mode='MinEculidean')
	kMeans.plotCluster(data.aligned_matrix_3D,kMeans.label)

	kMeans.fit(data.aligned_matrix_2D,distance_mode='Eculidean')
	kMeans.plotCluster(data.aligned_matrix_3D,kMeans.label)

	
	k_means = cluster.KMeans(n_clusters=num_cell)
	k_means.fit(data.aligned_matrix_2D)
	kMeans.distance_mode='sklearn'
	kMeans.plotCluster(data.aligned_matrix_3D,k_means.labels_)
	


if __name__ == "__main__":
	main()








