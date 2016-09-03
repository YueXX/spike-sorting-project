
import numpy as np
from Kmeans_spikeDetection import Kmeans_spikeDetection
from data_initialization_spikeSorting import data_initialization_spikeSorting
from sklearn import cluster, datasets
from matplotlib import pyplot as plt
import sys
import os

sys.dont_write_bytecode = True

	# set parameter
num_cell=2
num_electron=2
	
overlap_level1=300
overlap_level2=100

noise_level=20
data=data_initialization_spikeSorting(num_cell=num_cell,num_electron=num_electron,time=3000,delay=False)
