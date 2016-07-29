import numpy as np
import random as rand
from scipy.spatial import distance
from matplotlib.pyplot import cm 
from matplotlib import pyplot as plt



def init_centroids(X,num_cluster):

	num_point=X.shape[0]#num of points
	dim_point=X.shape[1]#dim of the points
	
	# Take initialize centers
	index_permutation=np.random.permutation(num_point)
	initial_center=np.zeros((num_cluster,dim_point))

	#return initial_center
	for index in range(num_cluster):
		initial_center[index]=X[index_permutation[index]]


	return initial_center


def k_means_distance(X,center_vectors):
	
	clusters_distance=distance.cdist(X,center_vectors,'euclidean',p=2)
	distance_label=clusters_distance.argmin(axis=1)

	return distance_label


def k_means_findCenter(X,num_cluster,distance_label):
	
	center_vectors=np.zeros((num_cluster,X.shape[1]))

	for index in range(0,num_cluster):
		cluster_vector=X[distance_label==index]
		
		number=cluster_vector.shape[0]
		# Get new center by averaging vectors in a certain group
		center_vectors[index]=1.0/number*np.sum(cluster_vector,axis=0)			

	return center_vectors



def k_means_spikeDetection(X,num_cluster,iterations):
	
	# Initialize spikes with lables
	initial_center=init_centroids(X,num_cluster)

	# Main algorithm:
	center_vectors=initial_center
	
	for ite in range(iterations):
		
		# Determine clusters by computing the Eculidean distance and label
		distance_label=k_means_distance(X,center_vectors)
		center_vectors=k_means_findCenter(X,num_cluster,distance_label)

		# Get new center by averaging each cluster

	return center_vectors,distance_label



def init_centroids_electronBlock(X,num_cluster):

	X=np.array(X)

	# here aligned_spikes is 3-D array. Each block(there are num_electron blocks)
	# is a num_spike*spike_sim matrix
	num_electron=X.shape[0]
	num_point=X.shape[1]
	point_dim=X.shape[2]


	# Initialize center of k-means clustering
	index_permutation=np.random.permutation(num_point)
	initial_center=np.zeros((num_electron,num_cluster,point_dim))

	# return randomly initialized centers
	for index in range(num_cluster):
		initial_center[:,index,:]=X[:,index_permutation[index],:]

	return initial_center



def k_means_SumElectron_distance(X,center_vectors):

	num_point=X.shape[1]
	num_electron=X.shape[0]
	num_cluster=center_vectors.shape[1]

	electron_distance=np.zeros((num_point,num_cluster))


	for i in range(num_electron):
		spike=X[i,:,:]
		center=center_vectors[i,:,:]		
		distance_singleEletron=distance.cdist(spike,center,'euclidean')
			
			# Sum over distance in each electron
		electron_distance=electron_distance+distance_singleEletron

	label=electron_distance.argmin(axis=1)

	return label




def k_means_MinEculidean_distance(X,center_vectors):

	num_point=X.shape[1]
	num_electron=X.shape[0]
	num_cluster=center_vectors.shape[1]
	
	label=[]
	distance_matrix=np.zeros((num_electron,num_cluster))

	for index in range(num_point):
		
		for index2 in range(num_electron):

			single_electron=X[index2,index,:]
			distance_matrix[index2]=distance.cdist([single_electron],center_vectors[index2,:,:])

		label_=np.unravel_index(distance_matrix.argmin(),distance_matrix.shape)
		number=int(label_[1])
		
		label.append(number)
		
	return label



def k_means_findCenter_block(X,num_cluster,label):
	label=np.array(label)
	num_electron=X.shape[0]
	point_dim=X.shape[2]

	center_vectors=np.zeros((num_electron,num_cluster,point_dim))

	for index in range(num_cluster):
		
		cluster_vector=X[:,label==index,:]
		number=cluster_vector.shape[1]
		center_vectors[:,index,:]=1.0/number*np.sum(cluster_vector,axis=1)			

	return center_vectors


def k_means_SumEculidean_spikeDetection(X,num_cluster,iterations):
	
	initial_center=init_centroids_electronBlock(X,num_cluster)
	center_vectors=initial_center.copy()

	for ite in range(iterations):
		
		label=k_means_SumElectron_distance(X,center_vectors)
		center_vectors=k_means_findCenter_block(X,num_cluster,label)
	
	return center_vectors,label
	


def k_means_MinEculidean_spikeDetection(X,num_cluster,iterations):
	initial_center=init_centroids_electronBlock(X,num_cluster)
	center_vectors=initial_center.copy()

	for ite in range(iterations):
		# To do 
		label=k_means_MinEculidean_distance(X,center_vectors)
		center_vectors=k_means_findCenter_block(X,num_cluster,label)

	return center_vectors,label




def classify_label(signal_matrix,label,name):

	num_electron=signal_matrix.shape[0]
	label=np.array(label)

	max_label=np.max(label)
	color=cm.rainbow(np.linspace(0,1,max_label+1))

	f,ax=plt.subplots(max_label+1,num_electron,sharex=True, sharey=True)
	
	for index in range(max_label+1):
		cluster=signal_matrix[:,label == index,:]
		for index2 in range(num_electron):
			for item in range(cluster.shape[1]):

				ax[index,index2].plot(cluster[index2,item],color=color[index])
				ax[index,index2].set_title('%s' %[index+1,index2+1])

	plt.savefig('image/%s.png'%name)

	#plt.show()	




class Kmeans_spikeDetection:

	def __init__(self,num_cluster,iterations):

		self.num_cluster=num_cluster
		self.iterations=iterations


	def fit(self,X,distance_mode):
		self.distance_mode=distance_mode
		if(distance_mode=='SumEculidean'):
			self.cluster_centers,self.label=k_means_SumEculidean_spikeDetection(X,self.num_cluster,self.iterations)
			

		elif(distance_mode=='MinEculidean'):
			self.cluster_centers,self.label=k_means_MinEculidean_spikeDetection(X,self.num_cluster,self.iterations)
			
		else:
			self.cluster_centers,self.label=k_means_spikeDetection(X,self.num_cluster,self.iterations)

		return self


	def plotCluster(self,X,label):

		classify_label(X,label,self.distance_mode)

		return self




	