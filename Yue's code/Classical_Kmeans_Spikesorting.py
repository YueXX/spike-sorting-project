from __future__ import division, print_function
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal
from sklearn import cluster
from peak_detect import detect_peaks
from scipy.spatial import distance




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



def k_means_spikeDetection(X,num_cluster,iterations=20):
	
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






















