

import numpy as np
import random as rand
from matplotlib import pyplot as plt
import random as rand
from matplotlib.pyplot import cm 


def spike_timeline_generator(time,interval_parameter,spike_len=100):
# Initilization
	var=0
	start_time=[]
	index=0
# Main loop to generate the time axis
	while var < time-spike_len:
		interval=rand.expovariate(1.0/interval_parameter)
		interval=int(interval)

		var=var+interval+spike_len	
		start_time.append(var)
		
		index=index+1

	start_time[-1]=time
	spike_time=np.array(start_time)
	
	x_axis=np.arange(0,time)
	y=np.zeros(time)
		
	for item in x_axis:
		if item in spike_time:
			y[item]=2



def waveform_generator(spike_timeline,shape_parameter,spike_len=100):
	# get shape parameters
	mu1=shape_parameter[0,0]
	mu2=shape_parameter[0,1]

	sigma1=shape_parameter[1,0]
	sigma2=shape_parameter[1,1]

	height1=shape_parameter[2,0]
	height2=shape_parameter[2,1]
	# Convert unit
	# start_time=np.array(spike_time)/unit
	time=spike_timeline[-1]
	
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
	index=len(spike_timeline)
	for item in spike_timeline[0:index-1]:
		spike_y[item:item+spike_len]=spike

	spike_y=np.array(spike_y)
	return spike_y



def multi_electrons_generator(num_electron,num_cell,time,delay=False,noise_level=0.01,overlap_level=1000,boolean=False,spike_len=100):

# set the boolean matrix for whether an electron can detect a single cell
	if(boolean!=False):
		boolean=np.random.randint(0,2,size=(num_electron,num_cell))
	else:
		boolean=np.ones((num_electron,num_cell))
	
	num_eachElectron=boolean.sum(axis=1)

# set the matrix that records the signal delays for each cell in different electrons
	matrix_delay=np.zeros([num_electron,num_cell])

# set the matrix for signal in different cell/electron
	signal_cell_electron=np.zeros((num_electron,num_cell,time))

# set the matrix for spike shape parameter in different cell, electron
	spike_shape_parameter=np.zeros((num_electron,num_cell,3,2))

# set the matrix for spike timeline information in different cell(different elecron is record in matrix_delay)
	spike_timeline_parameter=[]


	for j in range(num_cell):
		interval_parameter=overlap_level
		spike_timeline=spike_timeline_generator(time,interval_parameter,plot=False,spike_len=100)
		
		#print('timeline',len(spike_timeline))
		spike_timeline_parameter.append(spike_timeline)


		for i in range(num_electron):	
			if(delay==True):
				delay=np.random.randint(1,100)
				matrix_delay[i,j]=delay
			else:
				delay=0
			spike_timeline=spike_timeline+delay
			spike_timeline[-1]=time

			# set random spike shape parameter
			loc=np.random.permutation([-1,1])
			mu1=loc[0]*rand.randint(10,30)
			mu2=loc[1]*rand.randint(10,30)
			sigma1=rand.randint(1,20)
			sigma2=rand.randint(1,20)
			height1=rand.randint(300,500)
			height2=rand.randint(300,500)
			
			shape_parameter=np.array([[mu1,mu2],[sigma1,sigma2],[height1,height2]])

			spike_shape_parameter[i,j]=shape_parameter

			signal=waveform_generator(spike_timeline,shape_parameter,False,spike_len)*boolean[i,j]
			signal_cell_electron[i,j]=noise(signal,epsilon=(height1+height2)/2*noise_level)
			# get the matrix for different electrons
			matrix_electron=signal_cell_electron.sum(axis=1)


		# add plot 
	if(plot!=False):
		
		# plot for different cells in electrons
		color1=cm.rainbow(np.linspace(0,1,num_cell))

		f,ax=plt.subplots(num_electron,sharex=True, sharey=True)
	

		for i in range(num_electron):
			number=num_eachElectron[i]

			for j in range(num_cell):
				if(boolean[i,j]!=0):
					signal=np.array(signal_cell_electron[i,j])
					#signal=signal[0:10000]
				else:
					signal=0

				ax[i].plot(signal,color=color1[j])
				ax[i].set_title('Electron %s can receive signals from %s cells' %(i,number))
		#plt.savefig('image/SeperateSignalsOfElectron.png')
		plt.show()


		# plot for the compositions of cells signals in electrons
		f2,ax2=plt.subplots(num_electron,sharex=True, sharey=True)
		for i in range(num_electron):
			signal=np.array(matrix_electron[i])
			#signal=signal[0:10000]
			ax2[i].plot(signal,color='b')
			ax2[i].set_title('Signals of Electron %s' %(i))
		plt.savefig('image/ComposedSignalsOfElectron.png')
		#plt.show()


		# plot for spike shape in different electrons

		f3,ax3=plt.subplots(1)
	
		#spike_shape=[]
		for j in range(num_cell):
			spike_shape=[]


			for i in range(num_electron):
				parameter=spike_shape_parameter[i,j]

				mu1=parameter[0,0]
				mu2=parameter[0,1]
				sigma1=parameter[1,0]
				sigma2=parameter[1,1]
				height1=parameter[2,0]
				height2=parameter[2,1]

				spike_x=np.arange(-spike_len/2,spike_len/2)
				spike1=height1*np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
				spike2=height2*np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
				spike=spike1-spike2

				spike_shape=np.concatenate((spike_shape,spike),axis=0)

			ax3.plot(spike_shape)
			ax3.set_title('Original Spikes')
		
		plt.savefig('image/OriginalSpikes.png')


	return matrix_electron, boolean,signal_cell_electron,spike_timeline_parameter, spike_shape_parameter



def process_spike_multi(signal_matrix,timeline_matrix,threshold, window_height=2,spike_len=100):
	
	# get number of electron
	shape=signal_matrix.shape
	num_electron=shape[0]
	

	# take convolution of the first row of matrix
	signal1=signal_matrix[0]
	signal_abs=map(abs,signal1)
	
	weights = np.repeat(window_height, spike_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/spike_len

	# get information of spike location
	local_max=detect_peaks(convolution, mph=threshold, mpd=spike_len, show=False)

	# print(local_max-spike_len/2)
	# print('convolution',len(local_max))

	# Initialization
	
	m=len(local_max)
	n=spike_len
	


	# initialize label
	label=[]
	
	num_cell=len(timeline_matrix)


	# initialize empty 3D array for final aligned matrix

	final_matrix=np.zeros((num_electron,m,n))
	final_matrix_ecud=np.zeros((m,1))
	
	# initialize empty matrix for aligned matrix for each electron
	
	detected_spikes=np.zeros((m,n))

	# label each spike in one electron

	label_array=np.zeros(num_cell)
	


	for item in local_max:
		for index in range(num_cell):

			distance=[]
			distance=timeline_matrix[index]+spike_len/2-item

			distance=abs(distance)
			label_array[index]=np.amin(distance)


		sorted_array=sorted(label_array)
		if(sorted_array[1]<spike_len):
			new_label=num_cell
			label.append(new_label)
		else:
			label.append(np.argmin(label_array))


	# loop over each row of matrix_electron --- every electron
	for num in range(num_electron):
		
		# initialize index
		index=0
		
		signal=signal_matrix[num]

		# locate spike in electron signal and label them
		for item in local_max:
			detected_spikes[index]=signal[item-spike_len/2:item+spike_len/2]
			index=index+1


		# aligned detected spikes in one signel electron

		# random choose one row 
		k=rand.randint(0,m-1)

		# get the maximum location of this row
		max_location=detected_spikes[k].argmax(axis=0)

		# roll other rows according to this information
		for i in range(0,m):
			spike_max_location=detected_spikes[i].argmax(axis=0)
			distance=max_location-spike_max_location
			detected_spikes[i]=np.roll(detected_spikes[i],distance)

		# add the aligned to our final matrix
		final_matrix[num,:,:]=detected_spikes
		final_matrix_ecud=np.concatenate((final_matrix_ecud,detected_spikes),axis=1)
	
		#get rid of the first zeros in the matrix
	n=final_matrix_ecud.shape[1]
	final_matrix_ecud=final_matrix_ecud[:,1:n]

	return final_matrix,final_matrix_ecud


class data_initialization_spikeSorting:

	def __init__(self,num_cell,num_electron,time,delay):
		self.num_cell=num_cell
		self.num_electron=num_electron
		self.time=timeline_matrix
		self.delay=delay


	def 








