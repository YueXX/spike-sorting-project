
import numpy as np
import random as rand
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from collections import Counter
import sys
from scipy.spatial import distance

sys.dont_write_bytecode = True
sys.dont_write_bytecode = True


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def spike_timeline_generator(time, interval_parameter, spike_len):
	rand.seed()
	var = 0
	timeline = []
	index = 0
	# Main loop to generate the time axis
	while var < time - spike_len:
		interval = np.random.normal(interval_parameter, interval_parameter/4)

		interval = int(abs(interval))
		var = var + interval + spike_len
		timeline.append(var)

		index = index + 1

	timeline[-1] = time
	timeline = np.array(timeline)

	return timeline


# simple test function
# time_line=spike_timeline_generator(1000,100,100)


def signal_generator(timeline, shape_parameter, spike_len):
	# get shape parameters
	mu1 = shape_parameter[0, 0]
	mu2 = shape_parameter[0, 1]

	sigma1 = shape_parameter[1, 0]
	sigma2 = shape_parameter[1, 1]

	height1 = shape_parameter[2, 0]
	height2 = shape_parameter[2, 1]

	time = timeline[-1]

	# set the length for waveform
	cell_signal = np.zeros(time)

	spike_x = np.arange(-spike_len / 2, spike_len / 2)

	spike_left = height1 * np.exp(-np.power(spike_x / 1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike_right = height2 * np.exp(-np.power(spike_x / 1.0 - mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike = spike_left - spike_right

	# put spike into axis
	index = len(timeline)
	for item in timeline[0:index - 1]:
		cell_signal[item:item + spike_len] = spike

	cell_signal = np.array(cell_signal)
	return cell_signal


# simple test function
# shape_parameter=np.array([[0,-5],[10,10],[200,200]])
## 
# spike_len=100
# signal=signal_generator(time_line,shape_parameter,spike_len)
# plt.plot(signal)

def noise(signal, epsilon):
	length = len(signal)
	noise_vector = []
	for index in range(length):
		random = epsilon * rand.gauss(0, 2)
		noise_vector.append(random)

	noised_signal = signal + noise_vector
	return noised_signal


# noised_signal=noise(signal,20)

def multi_electrons_shape_generator(num_cell, num_electron):
	rand.seed()
	spike_shape_parameter = np.zeros((num_cell, num_electron, 3, 2))

	for i in range(num_cell):

		for j in range(num_electron):
			loc = np.random.permutation([-1, 1])
			mu1 = loc[0] * rand.randint(0, 30)
			mu2 = loc[1] * rand.randint(5, 30)
			sigma1 = rand.randint(1, 20)
			sigma2 = rand.randint(1, 20)
			height1 = rand.randint(300, 800)
			height2 = rand.randint(300, 800)

			shape_parameter = np.array([[mu1, mu2], [sigma1, sigma2], [height1, height2]])

			spike_shape_parameter[i, j] = shape_parameter

	return spike_shape_parameter


def multi_electrons_signal_generator(num_cell, num_electron, spike_shape_parameter, time, delay, overlap_level,
                                     noise_level, boolean, spike_len):
	rand.seed()
	# initialize cell with different delay in electrons
	delay_matrix = np.zeros((num_cell, num_electron))

	# initialize 3-D matrix to store signal in each cell of each electron
	signal_matrix = np.zeros((num_cell, num_electron, time))

	# initialize list to store timeline for each cell
	timeline_list = []
	num_spike = 0
	for i in range(num_cell):

		# Generate different timeline for different cell
		#interval_parameter = np.random.normal(overlap_level, overlap_level / 2)
		interval_parameter = np.random.normal(overlap_level/2, 2*overlap_level)
		

		interval_parameter = int(abs(interval_parameter))
		cell_timeline = spike_timeline_generator(time, interval_parameter, spike_len)

		# store timeline to list
		timeline_list.append(cell_timeline)
		num_spike = num_spike + len(cell_timeline) - 1

		# generate signal for each cell of each electron
		for j in range(num_electron):

			# if delay
			if (delay == True):
				delay = np.random.randint(1, 100)
				delay_matrix[i, j] = delay
			else:
				delay = 0

			cell_timeline = cell_timeline + delay
			cell_timeline[-1] = time

			# generate each signal
			signal = signal_generator(cell_timeline, spike_shape_parameter[i, j], spike_len)

			# decide if a cell is going to appear in an electron
			signal = signal * boolean[i, j]
			signal = noise(signal, noise_level)

			# store electron 
			signal_matrix[i, j] = signal

		# add each the signal of every cell in one electron 
		signal = signal_matrix.sum(axis=0)

	return signal, timeline_list, signal_matrix, delay_matrix, num_spike


def set_cell_electron(num_cell, num_electron, detect_range):
	# a function to generate the boolean function determining whether a
	# cell appears in a electron
	# detect_range:(0,1], 1 means all the cells can be detect by each electron

	# random seed
	rand.seed()
	# initialze boolean matrix
	boolean = np.zeros((num_cell, num_electron))

	# make sure that all cells appear in at least 1 electron
	for i in range(num_cell):
		choose_electron2 = np.random.permutation(num_electron)[0:1 + int(num_electron * detect_range)]
		boolean[i][choose_electron2] = 1

	return boolean


def convert_X_block(X, num_electron, spike_len):
	# convert 2D signal matrix to 3D
	num_point = X.shape[0]
	X_block = np.zeros((num_electron, num_point, spike_len))

	for i in range(num_electron):
		X_block[i, :, :] = X[:, i * spike_len:(i + 1) * spike_len]

	return X_block


def process_aligned_signal(signal, timeline_list, num_spike, spike_len):
	# initialization
	num_electron = signal.shape[0]
	num_cell = len(timeline_list)
	aligned_spikes = np.zeros((num_spike, spike_len * num_electron))

	original_label = []
	ite = 0

	for index in range(num_cell):
		spike_loc = timeline_list[index]

		spike_loc_noendpoint = spike_loc[0:-1]

		for location in spike_loc_noendpoint:
			aligned_spikes[ite] = signal[:, location:location + spike_len].flatten()
			# plt.plot(aligned_spikes[ite],color=color[index])
			ite = ite + 1
			original_label.append(index)
		# plt.show()


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
	aligned_spikes3D = convert_X_block(aligned_spikes, num_electron, spike_len)

	return aligned_spikes, aligned_spikes3D, original_label



def init_centroids_electronBlock(X, num_cluster):
	rand.seed()
	X = np.array(X)

	# here aligned_spikes is 3-D array. Each block(there are num_electron blocks)
	# is a num_spike*spike_sim matrix
	num_electron = X.shape[0]
	num_point = X.shape[1]
	point_dim = X.shape[2]

	# Initialize center of k-means clustering
	index_permutation = np.random.permutation(num_point)

	initial_center = np.zeros((num_electron, num_cluster, point_dim))

	# return randomly initialized centers
	for index in range(num_cluster):
		initial_center[:, index, :] = X[:, index_permutation[index], :]
	return initial_center


def k_means_MinEculidean_distance(X, center_vectors, num_electron):
	count = 0
	num_point = X.shape[1]

	# print(num_point*num_electron,'num_point')
	label = []

	distance_matrix = np.zeros((num_electron, num_cluster))

	for index in range(num_point):

		for index2 in range(num_electron):

			single_electron = X[index2, index, :]
			# compare spike in each electron of cell with correspoding center vectors
			#print(type(center_vectors))
			center_single_electron = center_vectors[index2, :, :]
			distance_matrix[index2] = distance.cdist([single_electron], center_single_electron)

			# check if the spike pass certain threshold
			if np.amax(single_electron) <= 200 and np.amin(single_electron >= -200):
				# if loc in low_energy_list:
				for index3 in range(num_cluster):
					if np.amax(center_vectors[index2, index3, :] <= 200) and np.amin(
									center_vectors[index2, index3, :] >= -200):
						count = count + 1
						distance_matrix[index2, index3] = 1000000

					# get the smallest number in the matrix
		label_ = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
		# get the cluster index of the matrix
		# print(distance_matrix)
		number = int(label_[1])

		# print(count,'count')

		label.append(number)
	# print(count,'count')
	return label


def k_means_findCenter_block(X, num_cluster, num_electron, label):
	label = np.array(label)
	point_dim = X.shape[2]

	center_vectors = np.zeros((num_electron, num_cluster, point_dim))

	for index in range(num_cluster):

		cluster_vector = X[:, label == index, :]
		number = cluster_vector.shape[1]
		# print(number,'of cluster',index)

		# in case of bad initial points we get one empty cluster
		if (number == 0):
			center_vectors = 'not valid'
			return center_vectors

		# print(np.sum(cluster_vector,axis=1),'sum')
		center_vectors[:, index, :] = 1.0 / number * np.sum(cluster_vector, axis=1)
	return center_vectors


def initialization_centers(X, num_cluster):
	initial_center = init_centroids_electronBlock(X, num_cluster)
	#print('outer_loop')
	label = k_means_MinEculidean_distance(X, initial_center, num_electron)
	center_vectors = k_means_findCenter_block(X, num_cluster, num_electron, label)

	return center_vectors

def initialization_centers_kmeans(X,num_cluster):
	initial_center=init_centroids(X,num_cluster)
	label = k_means_distance(X,initial_center)
	center_vectors = k_means_findCenter(X,num_cluster,label)
	return center_vectors



def EM_algorithm_min(X,center,num_electron,num_cluster,iterations):

	for ite in range(iterations):
		label = k_means_MinEculidean_distance(X, center, num_electron)
		center_vectors = k_means_findCenter_block(X, num_cluster, num_electron, label)
		
	return center_vectors,label


def EM_algorithm_sum(X,center,num_electron,num_cluster,iterations):

	for ite in range(iterations):
		label=k_means_SumElectron_distance(X,center)
		center_vectors= k_means_findCenter_block(X, num_cluster, num_electron, label)

	return center_vectors,label

def k_means_SumEculidean_spikeDetection(X,num_electron,num_cluster,iterations,kmeans_iter):
	center_vectors_list = []
	label_list = []
	kmeans_iterations=0

	while kmeans_iterations<kmeans_iter:

		center_vectors = initialization_centers(X, num_cluster)
		
		while type(center_vectors) == str:
			center_vectors = initialization_centers(X, num_cluster)
			#print(type(center_vectors),'type')
		center_vectors,label=EM_algorithm_sum(X,center_vectors,num_electron,num_cluster,iterations)
		
		#if one full algorithm is not coverging
		if type(center_vectors)==str:
			print('not converge')
			continue
		
		else:
			kmeans_iterations=kmeans_iterations+1
			center_vectors_list.append(center_vectors)
			label_list.append(label)

	return center_vectors_list, label_list




def k_means_MinEculidean_spikeDetection(X, num_electron, num_cluster, iterations, kmeans_iter):
	center_vectors_list = []
	label_list = []
	kmeans_iterations=0

	while kmeans_iterations<kmeans_iter:

		center_vectors = initialization_centers(X, num_cluster)
		
		while type(center_vectors) == str:
			center_vectors = initialization_centers(X, num_cluster)
			#print(type(center_vectors),'type')
		center_vectors,label=EM_algorithm_min(X,center_vectors,num_electron,num_cluster,iterations)
		
		#if one full algorithm is not coverging
		if type(center_vectors)==str:
			print('not converge')
			continue
		
		else:
			kmeans_iterations=kmeans_iterations+1
			center_vectors_list.append(center_vectors)
			label_list.append(label)

	return center_vectors_list, label_list




def k_means_SumElectron_distance(X,center_vectors):

	num_point=X.shape[1]
	num_electron=X.shape[0]
	num_cluster=center_vectors.shape[1]

	electron_distance=np.zeros((num_point,num_cluster))


	for i in range(num_electron):
		spike=X[i,:,:]
		center=center_vectors[i,:,:]		
		distance_singleEletron=distance.cdist(spike,center,'euclidean')
			
			# Sum over distance in each electron
		electron_distance=electron_distance+distance_singleEletron

	label=electron_distance.argmin(axis=1)

	return label


# def k_means_SumEculidean_spikeDetection(X, num_electron, num_cluster, iterations, kmeans_iter):
# 	center_vectors_list = []
# 	label_list = []

# 	for i in range(kmeans_iter):

# 		center_vectors = initialization_centers(X, num_cluster)
# 		while type(center_vectors) == str:
# 			center_vectors = initialization_centers(X, num_cluster)

# 		for ite in range(iterations):
# 			label = k_means_SumElectron_distance(X, center_vectors)
# 			center_vectors = k_means_findCenter_block(X, num_cluster, num_electron, label)

# 		center_vectors_list.append(center_vectors)
# 		label_list.append(label)

# 	return center_vectors_list, label_list



def init_centroids(X,num_cluster):

	rand.seed()
	num_point=X.shape[0]#num of points
	dim_point=X.shape[1]#dim of the points
	
	# Take initialize centers
	index_permutation=np.random.permutation(num_point)
	initial_center=np.zeros((num_cluster,dim_point))

	#return initial_center
	for index in range(num_cluster):
		initial_center[index]=X[index_permutation[index]]


	return initial_center


def k_means_distance(X,center_vectors):
	
	clusters_distance=distance.cdist(X,center_vectors,'euclidean',p=2)
	distance_label=clusters_distance.argmin(axis=1)

	return distance_label


def k_means_findCenter(X,num_cluster,distance_label):
	
	center_vectors=np.zeros((num_cluster,X.shape[1]))

	for index in range(0,num_cluster):
		cluster_vector=X[distance_label==index]
		
		number=cluster_vector.shape[0]
		if number==0:
			center_vectors='not valid'
			return center_vectors

		# Get new center by averaging vectors in a certain group
		center_vectors[index]=1.0/number*np.sum(cluster_vector,axis=0)			

	return center_vectors



def k_means_spikeDetection(X, num_electron, num_cluster, iterations, kmeans_iter):
	center_vectors_list = []
	label_list = []

	for i in range(kmeans_iter):

		center_vectors = initialization_centers_kmeans(X, num_cluster)
		while type(center_vectors) == str:
			center_vectors = initialization_centers_kmeans(X, num_cluster)

		for ite in range(iterations):
			label = k_means_distance(X, center_vectors)
			center_vectors = k_means_findCenter(X, num_cluster,label)

		center_vectors_list.append(center_vectors)
		label_list.append(label)

	return center_vectors_list, label_list



def evaluate_kmeans(num_cluster,label,predict_label_list,cluster_centers_list,mode):
	
	label=np.array(label)
	percentage=[]
	#predict_label=np.array(predict_label)
	
	num_point=len(label)
	# if(len(label)!=len(predict_label)):
	# 	print('there is something wrong with the len of label')
	# 	return 
	
	for i in range(len(predict_label_list)):
		count=0
		for index in range(num_cluster):
			label_signle_kmeans=np.array(predict_label_list[i])
			

			real_label=label[label_signle_kmeans==index]
			
			group=most_common(real_label)
			
			real_label=list(real_label)
			count=count+real_label.count(group)

		percentage.append(1.0*count/num_point)
	
	print('the right prediction of ',mode,'is',max(percentage))

	high_prediction=np.argmax(percentage)
	predict_label=predict_label_list[high_prediction]
	predict_center=cluster_centers_list[high_prediction]

	return predict_label,predict_center



def classify_label(signal_matrix,label,name):

	num_electron=signal_matrix.shape[0]
	label=np.asarray(label)

	max_label=np.max(label)
	color=cm.rainbow(np.linspace(0,1,max_label+1))

	f,ax=plt.subplots(max_label+1,num_electron,sharex=True, sharey=True)
	
	for index in range(max_label+1):
		cluster=signal_matrix[:,label == index,:]
		#print(cluster.shape,'cluster',index)
		for index2 in range(num_electron):
			
			for item in range(cluster.shape[1]):

				ax[index,index2].plot(cluster[index2,item],color=color[index])
				ax[index,index2].set_title('%s' %[index+1,index2+1])

	plt.savefig('image/%s.png'%name)

	#plt.show()	

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


def compare(center1,center2,center3,num_electron,num_cluster):
	
	color=cm.rainbow(np.linspace(0,1,5))
	f,ax=plt.subplots(num_cluster,num_electron,sharex=True, sharey=True)

	for index in range (num_cluster):
		for index2 in range(num_electron):

			ax[index,index2].plot(center1[index2,index],color=color[1])
	plt.savefig('image/min_center.png')


	for index in range (num_cluster):
		for index2 in range(num_electron):

			ax[index,index2].plot(center2[index2,index],color=color[2])
	plt.savefig('image/sum_center.png')


	for index in range (num_cluster):
		for index2 in range(num_electron):

			ax[index,index2].plot(center3[index2,index],color=color[3])

	plt.savefig('image/Eculidean_center.png')


num_cell = 6
num_electron = 6
time = 25000
delay = False
overlap_level = 100
noise_level = 0
num_cluster = num_cell
#boolean=set_cell_electron(num_cell,num_electron,detect_range=0.7)

boolean=np.identity(6)
#boolean=np.array([[1,0,0],[0,1,0],[0,0,1]])
spike_len = 100
plot_size=2000
spike_shape_parameter = multi_electrons_shape_generator(num_cell, num_electron)
signal, timeline_list, signal_matrix, delay_matrix, num_spike = multi_electrons_signal_generator(num_cell, num_electron,
                                                                                                 spike_shape_parameter,
                                                                                                 time, delay,
                                                                                                 overlap_level,
                                                                                               noise_level, boolean,
                                                                                                 spike_len)
plot_data(num_cell,num_electron,signal_matrix,signal,spike_shape_parameter,boolean,spike_len,plot_size)

aligned_spikes, aligned_spikes3D, original_label = process_aligned_signal(signal, timeline_list, num_spike, spike_len)
X = aligned_spikes3D
iterations = 40
kmeans_iter = 15

center_vectors_list, label_list = k_means_MinEculidean_spikeDetection(X, num_electron, num_cluster, iterations,
                                                                      kmeans_iter)
mode='MinEculidean'
predict_label1,predict_center1 = evaluate_kmeans(num_cluster,original_label,label_list,center_vectors_list,mode)
#classify_label(aligned_spikes3D,predict_label,mode)


center_vectors_list, label_list = k_means_SumEculidean_spikeDetection(X, num_electron, num_cluster, iterations,
                                                                      kmeans_iter)
mode = 'SumEculidean'
predict_label2,predict_center2= evaluate_kmeans(num_cluster, original_label, label_list, center_vectors_list, mode)
classify_label(aligned_spikes3D,predict_label1,mode)


X_kmeans=aligned_spikes

center_vectors_list, label_list=k_means_spikeDetection(X_kmeans,num_electron,num_cluster,iterations,kmeans_iter)
classify_label(aligned_spikes3D,predict_label2,mode)

mode = 'Eculidean'
predict_label3,predict_center3 = evaluate_kmeans(num_cluster, original_label, label_list, center_vectors_list, mode)
classify_label(aligned_spikes3D,predict_label3,mode)

# print(predict_center1[0,0].shape)
compare(predict_center1,predict_center2,predict_center3,num_electron,num_cluster)

# print('lala')



# questions:
# specific type of data
# try real data?






