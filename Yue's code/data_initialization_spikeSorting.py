
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
from peak_detect import detect_peaks
import sys
sys.dont_write_bytecode = True



def spike_timeline_generator(time,interval_parameter,spike_len):
# Initilization
	rand.seed()
	var=0
	timeline=[]
	index=0
# Main loop to generate the time axis
	while var < time-spike_len:
		interval=np.random.normal(interval_parameter,interval_parameter/4)
		
		interval=int(abs(interval))
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
			height1=rand.randint(300,800)
			height2=rand.randint(300,800)
			
			shape_parameter=np.array([[mu1,mu2],[sigma1,sigma2],[height1,height2]])

			spike_shape_parameter[i,j]=shape_parameter

	return spike_shape_parameter



def multi_electrons_signal_generator(num_cell,num_electron,spike_shape_parameter,time,delay,overlap_level,noise_level,boolean,spike_len):
	rand.seed()
	# initialize cell with different delay in electrons
	delay_matrix=np.zeros((num_cell,num_electron))
	
	# initialize 3-D matrix to store signal in each cell of each electron
	signal_matrix=np.zeros((num_cell,num_electron,time))
	
	# initialize list to store timeline for each cell
	timeline_list=[]
	num_spike=0
	for i in range(num_cell):

		# Generate different timeline for different cell
		interval_parameter=np.random.normal(overlap_level,overlap_level/2)
		interval_parameter=int(abs(interval_parameter))
		cell_timeline=spike_timeline_generator(time,interval_parameter,spike_len)	
		
		# store timeline to list
		timeline_list.append(cell_timeline)
		num_spike=num_spike+len(cell_timeline)-1
		
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
	

	return signal,timeline_list,signal_matrix,delay_matrix,num_spike


def set_cell_electron(num_cell,num_electron,detect_range):
	
	# a function to generate the boolean function determining whether a 
	# cell appears in a electron
	# detect_range:(0,1], 1 means all the cells can be detect by each electron

	# random seed
	rand.seed()
	# initialze boolean matrix
	boolean=np.zeros((num_cell,num_electron))

	# make sure that all cells appear in at least 1 electron
	for i in range(num_cell):
		choose_electron2=np.random.permutation(num_electron)[0:1+int(num_electron*detect_range)]
		boolean[i][choose_electron2]=1

	return boolean



def plot_data(num_cell,num_electron,signal_matrix,signal,spike_shape_parameter,boolean,spike_len,plot_size):

	
	color1=cm.rainbow(np.linspace(0,1,num_cell))

	num_eachElectron=boolean.sum(axis=0)

	f,ax=plt.subplots(num_electron,sharex=True, sharey=True)
	
	for i in range(num_electron):
		number=num_eachElectron[i]

		for j in range(num_cell):
			if(boolean[j,i]!=0):
				signal_block=np.array(signal_matrix[j,i])
				signal_block=signal_block[0:plot_size]
					#signal=signal[0:10000]
			else:
				signal_block=0

			ax[i].plot(signal_block,color=color1[j])
			ax[i].set_title('Electron %s can receive signals from %s cells' %(i,number))
		#plt.savefig('image/SeperateSignalsOfElectron.png')
	plt.show()



	f2,ax2=plt.subplots(num_electron,sharex=True, sharey=True)
	for i in range(num_electron):
		electron_signal=signal[i]
		electron_signal=electron_signal[0:plot_size]
		ax2[i].plot(electron_signal,color='b')
		ax2[i].set_title('Signals of Electron %s' %(i))
		
		#plt.savefig('image/ComposedSignalsOfElectron.png')
	plt.show()




	f3,ax3=plt.subplots(num_cell,num_electron,sharex=True,sharey=True)
	
	for i in range(num_cell):

		for j in range(num_electron):
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
			spike=spike*boolean[i,j]

			ax3[i,j].plot(spike,color=color1[i])
			ax3[i,j].set_title('Signals from cell %s'%(i))

	
	plt.show()
	#plt.savefig('image/OriginalSpikes.png')

def convert_X_block(X,num_electron,spike_len):
	#convert 2D signal matrix to 3D
	num_point=X.shape[0]
	X_block=np.zeros((num_electron,num_point,spike_len))
	for i in range(num_electron):
		X_block[i,:,:]=X[:,i*spike_len:(i+1)*spike_len]

	return X_block


def process_aligned_signal(signal,timeline_list,num_spike,spike_len):
	# initialization
	num_electron=signal.shape[0]
	num_cell=len(timeline_list)
	aligned_spikes=np.zeros((num_spike,spike_len*num_electron))
	label=[]
	color=cm.rainbow(np.linspace(0,1,num_cell))

	ite=0

	for index in range(num_cell):
		spike_loc=timeline_list[index]
		
		spike_loc=spike_loc[0:-1]
		for location in spike_loc:
			aligned_spikes[ite]=signal[:,location:location+spike_len].flatten()
			#plt.plot(aligned_spikes[ite],color=color[index])
			ite=ite+1
			label.append(index)
		#plt.show()

	# if we use precise location of spikes, we don't need to do this
	# choose one row as rolling benchmark
	# k=rand.randint(0,num_spike-1)
	# max_location=aligned_spikes[k].argmax(axis=0)

	# # roll other rows according to this information
	# for num in range(num_spike):
	# 	spike_max_location=aligned_spikes[num].argmax(axis=0)
	# 	distance=max_location-spike_max_location
	# 	aligned_spikes[num]=np.roll(aligned_spikes[num],distance)
		# plt.plot(aligned_spikes[num])
		# plt.show()
	aligned_spikes3D=convert_X_block(aligned_spikes,num_electron,spike_len)
	return aligned_spikes,aligned_spikes3D,label

##################################################################################
##################################################################################


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

	def signal_generator(self,overlap_level,noise_level,detect_range):
		
		self.boolean=set_cell_electron(self.num_cell,self.num_electron,detect_range)

		self.signal,self.timeline_list,self.signal_matrix,self.delay_matrix,self.num_spike= \
		multi_electrons_signal_generator(self.num_cell,self.num_electron,self.spike_shape_parameter,self.time,self.delay,overlap_level,noise_level,self.boolean,self.spike_len)

		return self


	def align_signal(self):
		#To do 
		return self



	def process_spike(self):

		self.signal_matrix,self.modified_signal_matrix,self.true_label=process_aligned_signal(self.signal,self.timeline_list,self.num_spike,self.spike_len)

		return self


	# def get_aligned_signal(self,threshold=80,window_height=2):
	# 	self.aligned_matrix_3D,self.aligned_matrix_2D,self.spike_loc=process_aligned_signal(self.signal,self.timeline_list,threshold,self.spike_len,window_height)
	# 	return self



	def plot(self):
		plot_data(self.num_cell,self.num_electron,self.signal_matrix,self.signal,self.spike_shape_parameter,self.boolean,self.spike_len,plot_size=3000)



# def process_aligned_signal(signal,timeline_list,num_spike,threshold,spike_len,window_height):
	

# 	# get number of electron
# 	shape=signal.shape
# 	# the first row is assumed to be have all cells to get the location
# 	# of spike 
# 	num_electron=shape[0]
# 	num_cell=len(timeline_list)


# 	# take convolution of the first row of matrix or not?...

# 	i=0
# 	spike1=signal[i]
# 	signal_abs=map(abs,spike1)
	

# 	weights = np.repeat(window_height, spike_len)
# 	convolution=np.convolve(weights,signal_abs,'same')
# 	convolution=convolution/spike_len

# 	# get information of spike location
# 	#spike_loc=detect_peaks(convolution, mph=threshold, mpd=spike_len, show=False)


# 	# Initialization
# 	#m=len(spike_loc)
# 	m=num_spike
# 	n=spike_len
	

# 	# initialize empty 3D array for final aligned matrix
# 	final_matrix=np.zeros((num_electron,m,n))
# 	final_matrix_ecud=np.zeros((m,1))
	
# 	# initialize empty matrix for aligned matrix for each electron
# 	detected_spikes=np.zeros((m,n))

# 	# loop over each row of matrix_electron --- every electron
# 	for num in range(num_electron):
		
# 		# initialize index
# 		index=0
		
# 		signal_electron=signal[num]

# 		# locate spike in electron signal and label them
# 		for item in spike_loc:
# 			detected_spikes[index]=signal_electron[item-spike_len/2:item+spike_len/2]
# 			index=index+1

# 		# aligned detected spikes in one signel electron

# 		# random choose one row 
# 		k=rand.randint(0,m-1)

# 		# get the maximum location of this row
# 		max_location=detected_spikes[k].argmax(axis=0)

# 		# roll other rows according to this information
# 		for i in range(0,m):
# 			spike_max_location=detected_spikes[i].argmax(axis=0)
# 			distance=max_location-spike_max_location
# 			detected_spikes[i]=np.roll(detected_spikes[i],distance)

# 		# add the aligned to our final matrix
# 		final_matrix[num,:,:]=detected_spikes
# 		final_matrix_ecud=np.concatenate((final_matrix_ecud,detected_spikes),axis=1)
	
# 		#get rid of the first zeros in the matrix
# 	n=final_matrix_ecud.shape[1]
# 	final_matrix_ecud=final_matrix_ecud[:,1:n]

# 	return final_matrix,final_matrix_ecud,spike_loc


# def label(num_cell,num_electron,timeline_list,spike_loc,boolean,spike_len):

# 	# initialize label,each item in label is the 
# 	# label for ith electron
	
# 	label=[]
	
# 	# label each spike in one electron
# 	label_array=np.zeros(num_cell)
	
	# for item in spike_loc:
	# 	for index in range(num_cell):

	# 		distance=[]
	# 		distance=timeline_list[index]+spike_len/2-item

	# 		distance=abs(distance)
	# 		label_array[index]=np.amin(distance)

	# 	sorted_array=sorted(label_array)
	# 	if(sorted_array[1]<spike_len):
	# 		new_label=num_cell
	# 		label.append(new_label)
	# 	else:
	# 		label.append(np.argmin(label_array))















