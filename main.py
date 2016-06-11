import numpy as np
import random as rand
from matplotlib import pyplot as plt


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

# plt.plot(d[0:1000])
# plt.show()
plt.plot(d[0:5000])
d=map(abs,d)
