import numpy as np
import random as rand
from matplotlib import pyplot as plt

#new
#################################################################################################
# The following code will generate simulated signals for spike sorting

# spike_generator will generate the timeline when the spike will appear
# waveform generator will generate the wave form according to the timeline

# Input: 
# time: the set total time for the signal 
# spike_len: the length of each spike(the cut-off length of spike simulated by Gaussian difference)

# mu,sigma is the array of 2 Gaussian parameters that we use to determine the shape
# of the spike by using Gaussian difference

# lambda: the parameter (mean) in exponential distribution, must be set in accordance with the unit
# for example, if unit is set to 0.1, then lambd cannot be set as 0.001

# unit: the unit time (can be set as 1, 0.1,0.01 etc), the smaller we set we get more finer
# shape of spike

# plot: true is to plot 

# Output:
# two vectors specifies the x-axis of the signal and the y-axis of the signal

def spike_generator(time, spike_len, interval_parameter):
# Initilization
	var=0
	start_time=[]
	index=0
# Change variable according to different unit
	# parameter_unit=interval_parameter/unit
	# spike_len_unit=spike_len/unit
	# time_unit=time/unit


# Main loop to generate the time axis
	while var < time:
		interval=rand.expovariate(1.0/interval_parameter)
		interval=int(interval)

		var=var+spike_len+interval		
		start_time.append(var)
		
		index=index+1

	start_time[-1]=time
	spike_time=np.array(start_time)
	

	return spike_time
	


#################################################################
def waveform_generator(spike_time,spike_len,shape_parameter):
	# get shape parameters
	mu1=shape_parameter[0,0]
	mu2=shape_parameter[0,1]

	sigma1=shape_parameter[1,0]
	sigma2=shape_parameter[1,1]

	height1=shape_parameter[2,0]
	height2=shape_parameter[2,1]

	# Convert unit
	# start_time=np.array(spike_time)/unit
	time=spike_time[-1]
	
	# time_unit=time/unit
	# spike_len_unit=spike_len/unit
	
	# set the length for waveform
	x=np.arange(time)
	y=np.zeros(time)
	spike_y=y.copy()

	
	
	# set for axis
	x_axis=x

	# draw the spikes
	spike_x=np.arange(-spike_len/2,spike_len/2)

	spike1=height1*np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike2=height2*np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike=spike1-spike2


	# put spike into axis
	index=len(spike_time)
	for item in spike_time[0:index-2]:
		spike_y[item:item+spike_len]=spike


	return x_axis,spike_y


#################################################################
# plot function for spike_time


def plot_spike(spike_time):
	time=spike_time[-1]
	
	x=np.arange(0,time)
	y=np.zeros(time)
		
	for item in x:
		if item in spike_time:
			y[item]=1

	return y


#################################################################
# Noise function: 
# adding noise to the signal

# input: 
# signal: original y-axis of the signal
# epsilon: control the degree of noise 
def noise(signal,epsilon):
	length=len(signal)
	noise_fun=[]
	for index in range(1,length+1):
		random=epsilon*rand.gauss(0, 2)
		noise_fun.append(random)
	
	output=signal+noise_fun
	return output


##################################################################
# Try for some data: 
# Try to simulate the compositions of signals from two cells


# time=1000
# spike_len=20
# lambd=70
# shape_parameter=np.array([[0,2.5],[2,3]])
# shape_parameter2=np.array([[0,4],[2,2]])



# a=spike_generator(time,spike_len,lambd)
# a2=spike_generator(time,spike_len,lambd)

# pl=plot_spike(a)

# #plt.plot(pl[0:50])

# #plt.show()

# b,c=waveform_generator(a,spike_len,shape_parameter)
# b2,c2=waveform_generator(a2,spike_len,shape_parameter2)

# d=c+c2
# d=noise(d,0.01)

# # plt.plot(d[0:1000])
# # plt.show()
# plt.plot(d[0:5000])
# d=map(abs,d)


# weights = np.repeat(1, spike_len/0.1)

# con=np.convolve(weights,d,'same')


# con=con/(50)


# # #print(con[0:200])
# plt.plot(con[0:5000])
# plt.show()


