# simple function to test k-means on 2 cells/2 electrons 


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
from data_initilization_2D import noise
from data_initilization_2D import plot_spike
#from k_means import detect_spike

############################################################
# Generate stimulated data
time=80000
spike_len=500

lambd1=1200
lambd2=800


shape_parameter=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter2=np.array([[0,-50],[40,20],[1000,1000]])
shape_parameter3=np.array([[0,30],[30,30],[1000,800]])
shape_parameter4=np.array([[0,-20],[25,30],[2000,1500]])

a=spike_timeline_generator(time,lambd1,False,spike_len)
a2=spike_timeline_generator(time,lambd2,False,spike_len)

c=waveform_generator(a,shape_parameter,spike_len)
c2=waveform_generator(a2,shape_parameter2,spike_len)

c3=waveform_generator(a,shape_parameter3,spike_len)
c4=waveform_generator(a2,shape_parameter4,spike_len)


signal1=c+c2
signal1=noise(signal1,5)

signal2=c3+c4
signal2=noise(signal2,5)

plt.plot(signal1[0:9000])
plt.show()

plt.plot(signal2[0:9000])
plt.show()





























