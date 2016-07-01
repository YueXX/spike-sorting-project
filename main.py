# test code

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

def main():
# generate data
	num_electron=2
	num_cell=3
	time=30000
	noise_level=0.01
	overlap_level=1000
	boolean=False
	plot=True
	threshold=100

	matrix_electron, boolean, spike_shape_parameter=multi_electrons_generator(num_electron,num_cell,time)



#process data

	a=process_spike_multi(matrix_electron,threshold)

	num_cluster=3
	interations=20

	# apply k-means
	center_vectors, classified_spikes=k_means_spikeDetection(a,num_cluster,interations)

	# plot result
	plot_kMeans_clusters(classified_spikes,center_vectors,num_cluster)


if __name__ == "__main__":
    main()








