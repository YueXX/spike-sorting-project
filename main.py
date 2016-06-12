from __future__ import division, print_function
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from peak_detect import detect_peaks

##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells

from data_initilization_2D import spike_generator
from data_initilization_2D import waveform_generator
from data_initilization_2D import noise
from data_initilization_2D import plot_spike



time=1000
spike_len=20
lambd=70
shape_parameter=np.array([[0,2.5],[2,3]])
shape_parameter2=np.array([[0,4],[2,2]])



a=spike_generator(time,spike_len,lambd)
a2=spike_generator(time,spike_len,lambd)

pl=plot_spike(a)

#plt.plot(pl[0:50])

#plt.show()

b,c=waveform_generator(a,spike_len,shape_parameter)
b2,c2=waveform_generator(a2,spike_len,shape_parameter2)

d=c+c2
d=noise(d,0.01)

#plt.plot(d[0:2000])
# plt.show()
#plt.plot(d[0:1000])
d_new=map(abs,d)


weights = np.repeat(2, 0.6*spike_len/0.1)

con=np.convolve(weights,d_new,'same')


con=con/(spike_len/0.1)

#a=argrelextrema(con, np.greater)
#plt.plot(con[0:2000])


a=detect_peaks(con, mph=0.4, mpd=100, show=False)

#print(a)


i=0
m=len(a)
n=spike_len/0.2

detect_spike=np.zeros(200)
for item in a:
	detect_spike=d[item-100:item+100]
	plt.plot(detect_spike)
	plt.show()

	i=i+1



#print(a)
#print(con[0:1000])

# plt.plot(con[0:1000])
#plt.show()


# scipy.signal.find_peaks_cwt

# signal.argrelmax(y_array, order=5)
