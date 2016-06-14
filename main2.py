from __future__ import division, print_function
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal
from sklearn import cluster
from peak_detect import detect_peaks


import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
#from peak_detect import detect_spike

from k_means import process_spike
from k_means import k_means_spikeDetection
##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells

from data_initilization_2D import spike_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import noise
from data_initilization_2D import plot_spike
#from k_means import detect_spike

############################################################
# Generate stimulated data
time=100000
spike_len=500

lambd1=4000
lambd2=4000

shape_parameter=np.array([[0,20],[30,30],[2000,2000]])
shape_parameter2=np.array([[0,-50],[40,20],[500,500]])

a=spike_generator(time,spike_len,lambd1)
a2=spike_generator(time,spike_len,lambd2)

b,c=waveform_generator(a,spike_len,shape_parameter)
b2,c2=waveform_generator(a2,spike_len,shape_parameter2)


# plt.plot(c[0:5000])
# plt.plot(c2[0:5000])

# plt.show()

d=c+c2
#d=noise(d,0.02)

plt.plot(d)
plt.show()

signal=d

signal_abs=map(abs,signal)
#signal_abs=np.array(signal)**2
	
# Step 2: take convolution of the absolute value
window_height=1

window_len=spike_len*0.3

weights = np.repeat(window_height, window_len)
convolution=np.convolve(weights,signal_abs,'same')
convolution=convolution/window_len
plt.plot(convolution[0:2500])
plt.show()

	# Step 3: find the indices of local maxima of the convolution
noise_level=20
local_max=detect_peaks(convolution, mph=noise_level*5, mpd=spike_len/2, show=True)


m=len(local_max)
n=spike_len
detected_spikes=np.zeros((m,n/2))
index=0
for item in local_max:
	detected_spikes[index]=signal[item-spike_len/4:item+spike_len/4]
	index=index+1


for i in range(20):
	plt.plot(detected_spikes[i])

plt.show()



