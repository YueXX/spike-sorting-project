
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
from peak_detect import detect_peaks


def spike_timeline_generator(time,interval_parameter,spike_len):
# Initilization
	var=0
	timeline=[]
	index=0
# Main loop to generate the time axis
	while var < time-spike_len:
		interval=rand.expovariate(1.0/interval_parameter)
		interval=int(interval)

		var=var+interval+spike_len	
		timeline.append(var)
		
		index=index+1

	timeline[-1]=time
	timeline=np.array(timeline)

	return timeline
	

def cell_signal_generator(timeline,shape_parameter,spike_len):
	# get shape parameters
	mu1=shape_parameter[0,0]
	mu2=shape_parameter[0,1]

	sigma1=shape_parameter[1,0]
	sigma2=shape_parameter[1,1]

	height1=shape_parameter[2,0]
	height2=shape_parameter[2,1]
	
	time=timeline[-1]
	
	# set the length for waveform
	cell_signal=np.zeros(time)
	
	spike_x=np.arange(-spike_len/2,spike_len/2)

	spike_left=height1*np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike_right=height2*np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike=spike_left-spike_right

	# put spike into axis
	index=len(timeline)
	for item in timeline[0:index-1]:
		cell_signal[item:item+spike_len]=spike

	cell_signal=np.array(cell_signal)
	return cell_signal


def noise(signal,epsilon):
	length=len(signal)
	noise_fun=[]
	for index in range(1,length+1):
		random=epsilon*rand.gauss(0, 2)
		noise_fun.append(random)
	
	output=signal+noise_fun
	return output



def multi_electrons_shape_generator(num_cell,num_electron):
	
	rand.seed()
	spike_shape_parameter=np.zeros((num_cell,num_electron,3,2))

	for i in range(num_cell):
		
		for j in range(num_electron):

			loc=np.random.permutation([-1,1])
			mu1=loc[0]*rand.randint(10,30)
			mu2=loc[1]*rand.randint(10,30)
			sigma1=rand.randint(1,20)
			sigma2=rand.randint(1,20)
			height1=rand.randint(300,500)
			height2=rand.randint(300,500)
			
			shape_parameter=np.array([[mu1,mu2],[sigma1,sigma2],[height1,height2]])

			spike_shape_parameter[i,j]=shape_parameter

	return spike_shape_parameter




def multi_electrons_signal_generator(num_cell,num_electron,spike_shape_parameter,time,delay,overlap_level,noise_level,boolean,spike_len):

	# initialize cell with different delay in electrons
	delay_matrix=np.zeros((num_cell,num_electron))
	
	# initialize 3-D matrix to store signal in each cell of each electron
	signal_matrix=np.zeros((num_cell,num_electron,time))
	
	# initialize list to store timeline for each cell
	timeline_list=[]

	for i in range(num_cell):

		# Generate different timeline for different cell
		interval_parameter=overlap_level
		cell_timeline=spike_timeline_generator(time,interval_parameter,spike_len)	
		
		# store timeline to list
		timeline_list.append(cell_timeline)
		
		# generate signal for each cell of each electron
		
		for j in range(num_electron):

			# if delay
			if(delay==True):
				delay=np.random.randint(1,100)
				delay_matrix[i,j]=delay
			else:
				delay=0

			cell_timeline=cell_timeline+delay
			cell_timeline[-1]=time
			

			# generate each signal
			signal=cell_signal_generator(cell_timeline,spike_shape_parameter[i,j],spike_len)

			# decide if a cell is going to appear in electron
			signal=signal*boolean[i,j]
			signal=noise(signal,noise_level)
			
			# store electron 
			signal_matrix[i,j]=signal

		# add each the signal of every cell in one electron 
		signal=signal_matrix.sum(axis=0)
	
	return signal,timeline_list,delay_matrix




def process_aligned_signal(signal,timeline_list,threshold,spike_len,window_height):
	
	# get number of electron
	shape=signal.shape
	num_electron=shape[0]
	num_cell=len(timeline_list)


	# take convolution of the first row of matrix
	signal1=signal[0]
	signal_abs=map(abs,signal1)
	

	weights = np.repeat(window_height, spike_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/spike_len

	# get information of spike location
	local_max=detect_peaks(convolution, mph=threshold, mpd=spike_len, show=False)

	# Initialization
	
	m=len(local_max)
	n=spike_len
	

	# initialize label
	label=[]
	

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
			distance=timeline_list[index]+spike_len/2-item

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
		
		signal_electron=signal[num]

		# locate spike in electron signal and label them
		for item in local_max:
			detected_spikes[index]=signal_electron[item-spike_len/2:item+spike_len/2]
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

	return final_matrix,final_matrix_ecud,label



class data_initialization_spikeSorting:

	def __init__(self,num_cell,num_electron,time,delay,spike_len=100):
		self.num_cell=num_cell
		self.num_electron=num_electron
		self.time=time
		self.delay=delay
		self.spike_len=spike_len


	def data_init(self):
		self.spike_shape_parameter=multi_electrons_shape_generator(self.num_cell,self.num_electron)

		return self

	def signal_generator(self,overlap_level,noise_level,boolean):
		self.signal,self.timeline_list,self.delay_matrix= \
		multi_electrons_signal_generator(self.num_cell,self.num_electron,self.spike_shape_parameter,self.time,self.delay,overlap_level,noise_level,boolean,self.spike_len)

		return self


	def align_signal():
		#To do 

		return self


	def process_aligned_signal(self,threshold=80,window_height=2):

		self.processed_matrix,self.processed_matrix_Ecu,self.label=process_aligned_signal(self.signal,self.timeline_list,threshold,self.spike_len,window_height=2,)
		
		return self














