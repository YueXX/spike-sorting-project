
import numpy as np
import random as rand
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from collections import Counter
import sys
from scipy.spatial import distance


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


def signal_generator(shape_parameter, spike_len):
	# get shape parameters
	mu1 = shape_parameter[0, 0]
	mu2 = shape_parameter[0, 1]

	sigma1 = shape_parameter[1, 0]
	sigma2 = shape_parameter[1, 1]

	height1 = shape_parameter[2, 0]
	height2 = shape_parameter[2, 1]

	spike_x = np.arange(-spike_len / 2, spike_len / 2)

	spike_left = height1 * np.exp(-np.power(spike_x / 1.0 - mu1, 2.) / (2 * np.power(sigma1, 2.)))
	spike_right = height2 * np.exp(-np.power(spike_x / 1.0 - mu2, 2.) / (2 * np.power(sigma2, 2.)))
	spike = spike_left - spike_right

	return spike



def noise(signal, epsilon):
	length = len(signal)
	noise_vector = []
	for index in range(length):
		random = epsilon * rand.gauss(0, 2)
		noise_vector.append(random)

	noised_signal = signal + noise_vector
	return noised_signal


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

num_cell=1
num_electron=3
spike_len=100

shapeparameter = multi_electrons_shape_generator(2, num_electron)

print(shapeparameter[0])
print(shapeparameter[1])


def aligned_spike_distance(shapeparameter,num_electron,spike_len):
	spike_matrix=np.zeros((2,num_electron,spike_len))

	for i in range(num_electron):
		shape1=shapeparameter[0,i]
		shape2=shapeparameter[1,i]

		spike1=signal_generator(shape1, spike_len)
		spike2=signal_generator(shape2, spike_len)

		spike1=noise(spike1,10)
		spike2=noise(spike2,10)

		spike_matrix[0,i]=spike1
		spike_matrix[1,i]=spike2

		dista=distance.cdist([spike1],[spike2])

		print(dista)

	f,ax=plt.subplots(num_electron,sharex=True, sharey=True)

	for j in range(num_electron):
			ax[j].plot(spike_matrix[0,j],color='r')
			ax[j].plot(spike_matrix[1,j],color='b')

	plt.show()

	return 0

aligned_spike_distance(shapeparameter,num_electron,spike_len)