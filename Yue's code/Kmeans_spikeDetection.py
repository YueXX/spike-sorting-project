import numpy as np
import random as rand
from scipy.spatial import distance
from matplotlib.pyplot import cm 
from matplotlib import pyplot as plt
from collections import Counter
import sys
sys.dont_write_bytecode = True

def init_centroids(X,num_cluster):

	rand.seed()
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



def k_means_spikeDetection(X,num_cluster,iterations,kmeans_iter):
	
	center_vectors_list=[]
	label_list=[]

	for i in range(kmeans_iter):
		# Initialize spikes with lables
		initial_center=init_centroids(X,num_cluster)

		# Main algorithm:
		center_vectors=initial_center
		
		for ite in range(iterations):

			# Determine clusters by computing the Eculidean distance and label
			distance_label=k_means_distance(X,center_vectors)
			center_vectors=k_means_findCenter(X,num_cluster,distance_label)

			# Get new center by averaging each cluster
		center_vectors_list.append(center_vectors)
		label_list.append(distance_label)

	return center_vectors_list,label_list



def init_centroids_electronBlock(X,num_cluster):
	
	rand.seed()
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
	print(initial_center.shape)
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


def k_means_SumElectron_dis(x,center_vectors,num_electron,num_cluster,spike_len):

	#initialize distance
	distance_init=np.zeros((1,num_cluster))

	# calculate distance for each electron
	for index in range(num_electron):
		x_electron=x[index*spike_len:(index+1)*spike_len]
		center_electron=center_vectors[:,index*spike_len:(index+1)*spike_len]
		
		distance_singleElectron=distance.cdist([x_electron],center_electron)
		distance_init=distance_init+distance_singleElectron

	label=np.argmin(distance_init,axis=1)
	label=int(label)

	return label




def k_means_ModifiedSum_spikeDetection(X,num_cluster,num_electron,iterations,kmeans_iter,spike_len):
	
	center_vectors_list=[]
	label_list=[]

	for i in range(kmeans_iter):
		
		# Initialize spikes with lables
		initial_center=init_centroids(X,num_cluster)
		# Main algorithm:
		center_vectors=initial_center
		
		for ite in range(iterations):
			distance_label=np.zeros(X.shape[0])

			for j in range(X.shape[0]):
				label=k_means_SumElectron_dis(X[j],center_vectors,num_electron,num_cluster,spike_len)
				distance_label[j]=int(label)

			# Determine clusters by computing the Eculidean distance and label
			center_vectors=k_means_findCenter(X,num_cluster,distance_label)
			# Get new center by averaging each cluster

		center_vectors_list.append(center_vectors)
		label_list.append(distance_label)
	return center_vectors_list,label_list


def detect_spikeEnergy(X,num_electron):
	num_point=X.shape[1]
	low_energy_list=[]

	for index in range(num_electron):
		for index2 in range(num_point):
			if(np.amax(X[index,index2,:])<=200):
				low_energy_index=[index,index2]
				low_energy_list.append(low_energy_index)

	#print(low_energy_list)
	return low_energy_list



def k_means_MinEculidean_distance(X,center_vectors,low_energy_list):

	#initialization
	count=0
	num_point=X.shape[1]
	num_electron=X.shape[0]
	num_cluster=center_vectors.shape[1]
	label=[]
	distance_matrix=np.zeros((num_electron,num_cluster))


	for index in range(num_point):

		for index2 in range(num_electron):
			single_electron=X[index2,index,:]
			
			# compare spike in each electron of cell with correspoding center vectors
			center_single_electron=center_vectors[index2,:,:]
			
			distance_matrix[index2]=distance.cdist([single_electron],center_single_electron)
			
			#check if the spike pass certain threshold
			loc=[index2,index]
            
			if loc in low_energy_list:
				for index3 in range(num_cluster):
					if(np.amax(center_vectors[index2,index3,:]<=200) and np.amin(center_vectors[index2,index3,:]>=-200)):
						count=count+1
						distance_matrix[index2,index3]=1000000

		# get the smallest number in the matrix
		label_=np.unravel_index(distance_matrix.argmin(),distance_matrix.shape)
		# get the cluster index of the matrix
		number=int(label_[1])
		label.append(number)
	print(distance_matrix)
	#print(count,'count')
	return label


def k_means_MinEculidean_distance2(X,center_vectors,low_energy_list):


	count=0
	num_point=X.shape[1]
	num_electron=X.shape[0]

	num_cluster=center_vectors.shape[1]
	
	#print(num_point*num_electron,'num_point')
	label=[]

	distance_matrix=np.zeros((num_electron,num_cluster))


	for index in range(num_point):

		for index2 in range(num_electron):

			single_electron=X[index2,index,:]
			# compare spike in each electron of cell with correspoding center vectors
			center_single_electron=center_vectors[index2,:,:]
			
			distance_matrix[index2]=distance.cdist([single_electron],center_single_electron)
			
			#check if the spike pass certain threshold
			#loc=[index2,index]

			if(np.amax(single_electron)<=200 and np.amin(single_electron>=-200)):
			#if loc in low_energy_list:
				for index3 in range(num_cluster):
					if(np.amax(center_vectors[index2,index3,:]<=200) and np.amin(center_vectors[index2,index3,:]>=-200)):

						count=count+1
						distance_matrix[index2,index3]=1000000
		
		# get the smallest number in the matrix
		label_=np.unravel_index(distance_matrix.argmin(),distance_matrix.shape)
		# get the cluster index of the matrix
		#print(distance_matrix)
		number=int(label_[1])
		
		#print(count,'count')

		label.append(number)
	#print(count,'count')
	return label



def k_means_findCenter_block(X,num_cluster,label):
	label=np.array(label)
	num_electron=X.shape[0]
	point_dim=X.shape[2]

	center_vectors=np.zeros((num_electron,num_cluster,point_dim))

	for index in range(num_cluster):
		
	
		cluster_vector=X[:,label==index,:]
		number=cluster_vector.shape[1]
		#print(number,'of cluster',index)

		#in case of bad initial points we get one empty cluster 
		if(number==0):
			center_vectors='not valid'
			return

		#print(np.sum(cluster_vector,axis=1),'sum')
		center_vectors[:,index,:]=1.0/number*np.sum(cluster_vector,axis=1)			
	return center_vectors


def k_means_SumEculidean_spikeDetection(X,num_cluster,iterations,kmeans_iter):
	
	# initialize to store different kmeans results

	center_vectors_list=[]
	label_list=[]
	for i in range(kmeans_iter):
		initial_center=init_centroids_electronBlock(X,num_cluster)
		center_vectors=initial_center.copy()

		for ite in range(iterations):

			label=k_means_SumElectron_distance(X,center_vectors)
			center_vectors=k_means_findCenter_block(X,num_cluster,label)
			
			if(type(center_vectors)==str):
				break

		center_vectors_list.append(center_vectors)
		label_list.append(label)

	return center_vectors_list,label_list




def k_means_MinEculidean_spikeDetection(X,num_electron,num_cluster,iterations,kmeans_iter):
	
	low_energy_list=detect_spikeEnergy(X,num_electron)
	center_vectors_list=[]
	label_list=[]
	
	for i in range(kmeans_iter):
		initial_center=init_centroids_electronBlock(X,num_cluster)
		center_vectors=initial_center.copy()

		for ite in range(iterations):

			label=k_means_MinEculidean_distance(X,center_vectors,low_energy_list)
			center_vectors=k_means_findCenter_block(X,num_cluster,label)

		center_vectors_list.append(center_vectors)
		label_list.append(label)

	return center_vectors_list,label_list




def k_means_MinEculidean_spikeDetection2(X,num_electron,num_cluster,iterations,kmeans_iter):
	low_energy_list=detect_spikeEnergy(X,num_electron)
	center_vectors_list=[]
	label_list=[]
	
	for i in range(kmeans_iter):
		initial_center=init_centroids_electronBlock(X,num_cluster)
		center_vectors=initial_center.copy()

		for ite in range(iterations):

			label=k_means_MinEculidean_distance2(X,center_vectors,low_energy_list)
			center_vectors=k_means_findCenter_block(X,num_cluster,label)

		center_vectors_list.append(center_vectors)
		label_list.append(label)

	return center_vectors_list,label_list






def classify_label(signal_matrix,label,name):

	num_electron=signal_matrix.shape[0]
	label=np.asarray(label)

	max_label=np.max(label)
	color=cm.rainbow(np.linspace(0,1,max_label+1))

	f,ax=plt.subplots(max_label+1,num_electron,sharex=True, sharey=True)
	
	for index in range(max_label+1):
		cluster=signal_matrix[:,label == index,:]
		#print(cluster.shape,'cluster',index)
		for index2 in range(num_electron):
			
			for item in range(cluster.shape[1]):

				ax[index,index2].plot(cluster[index2,item],color=color[index])
				ax[index,index2].set_title('%s' %[index+1,index2+1])

	#plt.savefig('image/%s.png'%name)

	plt.show()	


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def evaluate_kmeans(num_cluster,label,predict_label_list,cluster_centers_list,mode):
	
	label=np.array(label)
	percentage=[]
	#predict_label=np.array(predict_label)
	
	num_point=len(label)
	# if(len(label)!=len(predict_label)):
	# 	print('there is something wrong with the len of label')
	# 	return 
	
	for i in range(len(predict_label_list)):
		count=0
		for index in range(num_cluster):
			label_signle_kmeans=np.array(predict_label_list[i])
			

			real_label=label[label_signle_kmeans==index]
			
			group=Most_Common(real_label)
			
			real_label=list(real_label)
			count=count+real_label.count(group)

		percentage.append(1.0*count/num_point)
	
	print('the right prediction of ',mode,'is',max(percentage))

	high_prediction=np.argmax(percentage)
	predict_label=predict_label_list[high_prediction]
	predict_center=cluster_centers_list[high_prediction]

	return predict_label,predict_center


def plot_center(center_vectors,num_cluster,num_electron):

	color=cm.rainbow(np.linspace(0,1,num_cluster))

	f,ax=plt.subplots(num_cluster,num_electron,sharex=True, sharey=True)
	
	for index in range(num_cluster):
		#print(cluster.shape,'cluster',index)
		for index2 in range(num_electron):
			
			ax[index,index2].plot(center_vectors[index2,index,:],color=color[index])
			
	#plt.savefig('image/%s.png'%name)

	plt.show()	

			
		
class Kmeans_spikeDetection:

	def __init__(self,num_cluster,num_electron,iterations):

		self.num_cluster=num_cluster
		self.iterations=iterations
		self.num_electron=num_electron

	def fit(self,X,distance_mode,kmeans_iter=10):
		self.distance_mode=distance_mode
		if(distance_mode=='SumEculidean'):
			self.cluster_centers_list,self.predict_label_list=k_means_SumEculidean_spikeDetection(X,self.num_cluster,self.iterations,kmeans_iter)
			#self.cluster_center_list,self.predict_label_list=k_means_ModifiedSum_spikeDetection(X,self.num_cluster,self.num_electron,self.iterations,kmeans_iter,spike_len=100)
	
		elif(distance_mode=='MinEculidean'):

			self.cluster_centers_list,self.predict_label_list=k_means_MinEculidean_spikeDetection(X,self.num_electron,self.num_cluster,self.iterations,kmeans_iter)
		else:
			self.cluster_centers_list,self.predict_label_list=k_means_spikeDetection(X,self.num_cluster,self.iterations,kmeans_iter)

		return self


	def min_threshold(self,X,distance_mode,kmeans_iter=1):
		self.cluster_centers_list,self.predict_label_list=k_means_MinEculidean_spikeDetection2(X,self.num_electron,self.num_cluster,self.iterations,kmeans_iter)
	

	def evaluate(self,true_label,predict_label_list,mode):
		self.predict_label,self.predict_center=evaluate_kmeans(self.num_cluster,true_label,predict_label_list,self.cluster_centers_list,mode)
		
		return self


	def plotCluster(self,X,label):

		classify_label(X,label,self.distance_mode)

		return self


	def plotCenter(self):
		plot_center(self.predict_center,self.num_cluster,self.num_electron)

		return self





	