# Signal Simulator
# The returned signal is 2 vectors: x-axis/y-axis

import numpy as np
import random as rand
from matplotlib import pyplot as plt

#################################################################################################
# Signal simulator function: this function will simulate signal of one single cell 
# the random interval between two consecutive spike is in Exponential distribution 


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

def signal_simulator(time, spike_len, mu,sigma,lambd,unit=0.1, plot=False):

	var=0
	start_time=[]

	index=0
	lambd_unit=lambd/unit
	spike_len_unit=spike_len/unit
	time_unit=time/unit

	mu1=mu[0]
	mu2=mu[1]
	sigma1=sigma[0]
	sigma2=sigma[1]

	while var < time_unit:
		interval=rand.expovariate(1.0/lambd_unit)
		
		interval=int(interval)

		var=var+spike_len_unit+interval		
		start_time.append(var)
		
		index=index+1

	start_time[-1]=time_unit
	start_time1=np.array(start_time)*unit
	
	

	x=np.arange(time_unit)
	y=np.zeros(time_unit)
	spike_y=y.copy()

	
	for item in x:
		if item in start_time:
			y[item]=1

	x_axis=x*unit

	spike_x=np.arange(-spike_len_unit/2,spike_len_unit/2)

	spike1=np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike2=np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike=spike1-spike2


	index=len(start_time)
	for item in start_time[0:index-2]:
		spike_y[item:item+spike_len_unit]=spike


	
	# plot
	if(plot!=False):
		plt.plot(x_axis,spike_y)
		plt.show()



	return x_axis,spike_y


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


x1,y1=signal_simulator(300,6,[0,3],[2,2],2,0.1,False)
x2,y2=signal_simulator(300,6,[0,4],[2,1],2,0.1,False)


y=y1+y2

y_n=noise(y,0.1)
plt.plot(x1,y)

plt.show()

plt.plot(x1,y_n)
plt.show()



