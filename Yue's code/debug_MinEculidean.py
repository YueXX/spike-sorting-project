#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 23:46:13 2016

@author: starry1990
"""

import numpy as np
import random as rand
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from peak_detect import detect_peaks
import sys

sys.dont_write_bytecode = True
from scipy.spatial import distance


def spike_timeline_generator(time, interval_parameter, spike_len):
	# Initilization
	rand.seed()
	var = 0
	timeline = []
	index = 0
	# Main loop to generate the time axis
	while var < time - spike_len:
		interval = np.random.normal(interval_parameter, interval_parameter / 4)

		interval = int(abs(interval))
		var = var + interval + spike_len
		timeline.append(var)

		index = index + 1

	timeline[-1] = time
	timeline = np.array(timeline)

	return timeline


# simple test function
# time_line=spike_timeline_generator(1000,100,100)


def signal_generator(timeline, shape_parameter, spike_len):
	# get shape parameters
	mu1 = shape_parameter[0, 0]
	mu2 = shape_parameter[0, 1]

	sigma1 = shape_parameter[1, 0]
	sigma2 = shape_parameter[1, 1]

	height1 = shape_parameter[2, 0]
	height2 = shape_parameter[2, 1]

	time = timeline[-1]

	# set the length for waveform
	cell_signal = np.zeros(time)

	spike_x = np.arange(-spike_len / 2, spike_len / 2)

	spike_left = height1 * np.exp(-np.power(spike_x / 1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike_right = height2 * np.exp(-np.power(spike_x / 1.0 - mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike = spike_left - spike_right

	# put spike into axis
	index = len(timeline)
	for item in timeline[0:index - 1]:
		cell_signal[item:item + spike_len] = spike

	cell_signal = np.array(cell_signal)
	return cell_signal


# simple test function
# shape_parameter=np.array([[0,-5],[10,10],[200,200]])
## 
# spike_len=100
# signal=signal_generator(time_line,shape_parameter,spike_len)
# plt.plot(signal)

def noise(signal, epsilon):
	length = len(signal)
	noise_vector = []
	for index in range(length):
		random = epsilon * rand.gauss(0, 2)
		noise_vector.append(random)

	noised_signal = signal + noise_vector
	return noised_signal


# noised_signal=noise(signal,20)

def multi_electrons_shape_generator(num_cell, num_electron):
	rand.seed()
	spike_shape_parameter = np.zeros((num_cell, num_electron, 3, 2))

	for i in range(num_cell):

		for j in range(num_electron):
			loc = np.random.permutation([-1, 1])
			mu1 = loc[0] * rand.randint(0, 30)
			mu2 = loc[1] * rand.randint(5, 30)
			sigma1 = rand.randint(1, 20)
			sigma2 = rand.randint(1, 20)
			height1 = rand.randint(300, 800)
			height2 = rand.randint(300, 800)

			shape_parameter = np.array([[mu1, mu2], [sigma1, sigma2], [height1, height2]])

			spike_shape_parameter[i, j] = shape_parameter

	return spike_shape_parameter


def multi_electrons_signal_generator(num_cell, num_electron, spike_shape_parameter, time, delay, overlap_level,
									 noise_level, boolean, spike_len):
	rand.seed()
	# initialize cell with different delay in electrons
	delay_matrix = np.zeros((num_cell, num_electron))

	# initialize 3-D matrix to store signal in each cell of each electron
	signal_matrix = np.zeros((num_cell, num_electron, time))

	# initialize list to store timeline for each cell
	timeline_list = []
	num_spike = 0
	for i in range(num_cell):

		# Generate different timeline for different cell
		interval_parameter = np.random.normal(overlap_level, overlap_level / 2)
		interval_parameter = int(abs(interval_parameter))
		cell_timeline = spike_timeline_generator(time, interval_parameter, spike_len)

		# store timeline to list
		timeline_list.append(cell_timeline)
		num_spike = num_spike + len(cell_timeline) - 1

		# generate signal for each cell of each electron
		for j in range(num_electron):

			# if delay
			if (delay == True):
				delay = np.random.randint(1, 100)
				delay_matrix[i, j] = delay
			else:
				delay = 0

			cell_timeline = cell_timeline + delay
			cell_timeline[-1] = time

			# generate each signal
			signal = signal_generator(cell_timeline, spike_shape_parameter[i, j], spike_len)

			# decide if a cell is going to appear in an electron
			signal = signal * boolean[i, j]
			signal = noise(signal, noise_level)

			# store electron 
			signal_matrix[i, j] = signal

		# add each the signal of every cell in one electron 
		signal = signal_matrix.sum(axis=0)

	return signal, timeline_list, signal_matrix, delay_matrix, num_spike


def set_cell_electron(num_cell, num_electron, detect_range):
	# a function to generate the boolean function determining whether a
	# cell appears in a electron
	# detect_range:(0,1], 1 means all the cells can be detect by each electron

	# random seed
	rand.seed()
	# initialze boolean matrix
	boolean = np.zeros((num_cell, num_electron))

	# make sure that all cells appear in at least 1 electron
	for i in range(num_cell):
		choose_electron2 = np.random.permutation(num_electron)[0:1 + int(num_electron * detect_range)]
		boolean[i][choose_electron2] = 1

	return boolean


def convert_X_block(X, num_electron, spike_len):
	# convert 2D signal matrix to 3D
	num_point = X.shape[0]
	X_block = np.zeros((num_electron, num_point, spike_len))

	for i in range(num_electron):
		X_block[i, :, :] = X[:, i * spike_len:(i + 1) * spike_len]

	return X_block


def process_aligned_signal(signal, timeline_list, num_spike, spike_len):
	# initialization
	num_electron = signal.shape[0]
	num_cell = len(timeline_list)
	aligned_spikes = np.zeros((num_spike, spike_len * num_electron))

	original_label = []
	ite = 0

	for index in range(num_cell):
		spike_loc = timeline_list[index]

		spike_loc_noendpoint = spike_loc[0:-1]

		for location in spike_loc_noendpoint:
			aligned_spikes[ite] = signal[:, location:location + spike_len].flatten()
			# plt.plot(aligned_spikes[ite],color=color[index])
			ite = ite + 1
			original_label.append(index)
		# plt.show()


		# if we use precise location of spikes, we don't need to do this
		# choose one row as rolling benchmark
		# k=rand.randint(0,num_spike-1)
		# max_location=aligned_spikes[k].argmax(axis=0)

		# # roll other rows according to this information
		# for num in range(num_spike):
		# 	spike_max_location=aligned_spikes[num].argmax(axis=0)
		# 	distance=max_location-spike_max_location
		# 	aligned_spikes[num]=np.roll(aligned_spikes[num],distance)
		# plt.plot(aligned_spikes[num])
		# plt.show()
	aligned_spikes3D = convert_X_block(aligned_spikes, num_electron, spike_len)

	return aligned_spikes, aligned_spikes3D, original_label


def init_centroids_electronBlock(X, num_cluster):
	rand.seed()
	X = np.array(X)

	# here aligned_spikes is 3-D array. Each block(there are num_electron blocks)
	# is a num_spike*spike_sim matrix
	num_electron = X.shape[0]
	num_point = X.shape[1]
	point_dim = X.shape[2]

	# Initialize center of k-means clustering
	index_permutation = np.random.permutation(num_point)

	initial_center = np.zeros((num_electron, num_cluster, point_dim))

	# return randomly initialized centers
	for index in range(num_cluster):
		initial_center[:, index, :] = X[:, index_permutation[index], :]
	return initial_center, index_permutation


def k_means_MinEculidean_distance(X, center_vectors, num_electron):
	count = 0
	num_point = X.shape[1]

	# print(num_point*num_electron,'num_point')
	label = []

	distance_matrix = np.zeros((num_electron, num_cluster))

	for index in range(num_point):

		for index2 in range(num_electron):

			single_electron = X[index2, index, :]
			# compare spike in each electron of cell with correspoding center vectors
			center_single_electron = center_vectors[index2, :, :]
			distance_matrix[index2] = distance.cdist([single_electron], center_single_electron)

			# check if the spike pass certain threshold
			if np.amax(single_electron) <= 200 and np.amin(single_electron >= -200):
				# if loc in low_energy_list:
				for index3 in range(num_cluster):
					if np.amax(center_vectors[index2, index3, :] <= 200) and np.amin(
									center_vectors[index2, index3, :] >= -200):
						count = count + 1
						distance_matrix[index2, index3] = 1000000

					# get the smallest number in the matrix
		label_ = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
		# get the cluster index of the matrix
		# print(distance_matrix)
		number = int(label_[1])

		# print(count,'count')

		label.append(number)
	# print(count,'count')
	return label


def k_means_findCenter_block(X, num_cluster, num_electron, label):
	label = np.array(label)
	point_dim = X.shape[2]

	center_vectors = np.zeros((num_electron, num_cluster, point_dim))

	for index in range(num_cluster):

		cluster_vector = X[:, label == index, :]
		number = cluster_vector.shape[1]
		# print(number,'of cluster',index)

		# in case of bad initial points we get one empty cluster
		if (number == 0):
			center_vectors = 'not valid'
			return center_vectors

		# print(np.sum(cluster_vector,axis=1),'sum')
		center_vectors[:, index, :] = 1.0 / number * np.sum(cluster_vector, axis=1)
	return center_vectors


def initialization_centers(X, num_cluster):
	initial_center, index_permutation = init_centroids_electronBlock(X, num_cluster)
	label = k_means_MinEculidean_distance(X, initial_center, num_electron)
	center_vectors = k_means_findCenter_block(X, num_cluster, num_electron, label)

	return center_vectors


def k_means_MinEculidean_spikeDetection(X, num_electron, num_cluster, iterations, kmeans_iter):
	center_vectors_list = []
	label_list = []

	for i in range(kmeans_iter):

		center_vectors = initialization_centers(X, num_cluster)
		while type(center_vectors) == str:
			center_vectors = initialization_centers(X, num_cluster)

		for ite in range(iterations):
			label = k_means_MinEculidean_distance(X, center_vectors, num_electron)
			center_vectors = k_means_findCenter_block(X, num_cluster, num_electron, label)

		center_vectors_list.append(center_vectors)
		label_list.append(label)

	return center_vectors_list, label_list


num_cell = 2
num_electron = 2
time = 1000
delay = False
overlap_level = 100
noise_level = 0
num_cluster = num_cell
# boolean=set_cell_electron(num_cell,num_electron,detect_range=0.3)
boolean = np.array([[1, 0], [0, 1]])
spike_len = 100

spike_shape_parameter = multi_electrons_shape_generator(num_cell, num_electron)
signal, timeline_list, signal_matrix, delay_matrix, num_spike = multi_electrons_signal_generator(num_cell, num_electron,
																								 spike_shape_parameter,
																								 time, delay,
																								 overlap_level,
																								 noise_level, boolean,
																								 spike_len)
aligned_spikes, aligned_spikes3D, original_label = process_aligned_signal(signal, timeline_list, num_spike, spike_len)
X = aligned_spikes3D
iterations = 2
kmeans_iter = 2

center_vectors_list, label_list = k_means_MinEculidean_spikeDetection(X, num_electron, num_cluster, iterations,
																	  kmeans_iter)
