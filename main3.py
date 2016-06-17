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
#from peak_detect import detect_spike

from k_means import process_spike
from k_means import k_means_spikeDetection
from k_means import plot_kMeans_clusters
##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells

from data_initilization_2D import spike_timeline_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import spike_shape_generator

from data_initilization_2D import noise
from data_initilization_2D import plot_spike
from data_initilization_2D import multi_electrons_generator
##################################################################

x=spike_timeline_generator(1000,20,False)

print(x)


shape_parameter=np.array([[-50,40],[30,30],[2000,2000]])


num_cell=3
num_electron=2

time=2500
mat=multi_electrons_generator(num_cell,num_electron,time)

#print(a*4)
#print(mat)
a=(mat[0,:])
plt.plot(a)
plt.show()







