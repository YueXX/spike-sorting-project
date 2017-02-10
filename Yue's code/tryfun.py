import numpy as np
import random as rand
from scipy.spatial import distance

from matplotlib.pyplot import cm 
from matplotlib import pyplot as plt

from data_initialization_spikeSorting import data_initialization_spikeSorting

from trymain import tryman
tryman(4)
# x=np.array([[[1,2,3],[2,3,4]],[[3,4,5],[4,5,6]]])
# print(np.argmax(x))
# mu, sigma = 0, 1 # mean and standard deviation
# s = np.random.normal(mu, sigma)
# print(s)
# spike_len=100



# def noise(signal,epsilon):
# 	length=len(signal)
# 	noise_fun=[]
# 	for index in range(1,length+1):
# 		random=epsilon*rand.gauss(0, 2)
# 		noise_fun.append(random)
	
# 	output=signal+noise_fun
# 	return output

# signal=100*np.ones(100)

# signal=noise(signal,20)
# signal_abs=map(abs,signal)
# print(np.mean(signal_abs))
# plt.plot(signal_abs)

# spike_len=100
# mu1=20
# mu2=30
# sigma1=20
# sigma2=10
# height1=500
# height2=500

# spike_x=np.arange(-spike_len/2,spike_len/2)

# spike_left=height1*np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
# spike_right=height2*np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
# spike=spike_left-spike_right

# spike=noise(spike,20)




# spike=map(abs,spike)

# print(np.mean(spike))

# plt.plot(spike)
# plt.show()

# window_height=1
# spike_len=100

# #plt.plot(signal_abs)
# # #plt.show()

# weights = np.repeat(window_height, spike_len)
# convolution=np.convolve(weights,signal_abs,'same')
# convolution=convolution/spike_len

# plt.plot(convolution)

# convolution2=np.convolve(weights,spike,'same')
# convolution2=convolution2/spike_len
# plt.plot(convolution2)
# plt.show()






# # height1=100
# # height2=100
# # sigma1=10
# # sigma2=15
# # mu1=10
# # mu2=30
# # x=[]
# # x.append(1)
# # x.append(2)


# X=np.ones((100))
# y=np.zeros((10,100))
# a=distance.cdist([X],y)
# print(type(X))
# print(X.shape)
# print(y.shape)
# print(a)
# spike_x=np.arange(-spike_len/4,spike_len/4*3)

# spike_left=height1*np.exp(-np.power(spike_x/1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
# spike_right=height2*np.exp(-np.power(spike_x/1.0- mu2, 2.) / (2 * np.power(sigma2, 2.)))
# spike=spike_left-spike_right


# def noise(signal,epsilon):
# 	length=len(signal)
# 	noise_fun=[]
# 	for index in range(1,length+1):
# 		random=epsilon*rand.gauss(0, 2)
# 		noise_fun.append(random)
	
# 	output=signal+noise_fun
# 	return output



# spike1=noise(spike,1)
# spike2=noise(spike,1)

# spike3=noise(np.zeros(spike_len),1)
# spike4=noise(np.zeros(spike_len),1)

# a=np.linalg.norm(spike1-spike2)
# b=np.linalg.norm(spike3-spike4)
# print(a)
# print(b)

# num_cell=6
# num_electron=7
# boolean=np.zeros((num_cell,num_electron))
	
# 	# make sure that all cells appear in at least 1 electron
# choose_electron=np.random.permutation(num_electron)[0:num_cell]

# detect_range=0.5

# for i in range(num_cell):
# 	boolean[i,choose_electron[i]]=1
# 	choose_electron2=np.random.permutation(num_electron)[0:int(num_electron*detect_range)]
	
# 	print(choose_electron2)
# 	boolean[i][choose_electron2]=1

# print(boolean)





# x=np.array([[1,2,3,7],[2,3,4,7],[3,4,5,7]])

# y=x.reshape(3,2,2)

#print(y[:,1,:])

# # shape=[]
# x=np.array([[[1,2,3],[3,4,5]],[[4,5,6],[5,6,7]]])

# for i in range(2):
# 	print(x[:,i,:])


# c=np.where(centers == np.min(centers))
# index=(np.unravel_index(centers.argmin(),centers.shape))
# index=np.array(index)


# x=np.array([[1,2,3,6],[3,4,5,7],[5,6,7,8],[7,8,9,0]])

# print(x)
# print(x[:,1:3])
# y=x.reshape((2,4,2))
# print(x)
# print(y)
# print(y.shape)


