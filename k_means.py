import numpy as np
import random as rand
from matplotlib import pyplot as plt
from peak_detect import detect_peaks

#########################################################
# The following code apply K-means algorithms to signal
# by three steps: spike detection, alignment of spikes and k-means algorithm


# Question 1: we have to manually set the window length/convolution window,  
# since we use convolution window to find spike, the convolution result may not give
# us the best estimate

# Question 2: align spikes 

#########################################################
# Detect_spike function will detect spikes in a signal:
# input: 
# signal: the input signal
# window_len: the manually set length for window in convolution
# window_height: the manually set height for window in convolution
# noise_level: the lower bound parameter in the find local maxima function

# output: the unaligned spike in a 2-D array detected_spikes


def process_spike(signal, window_len, noise_level, window_height=2):
	
	################################################
	# Step 1: take the absolute value of signal
	signal_abs=map(abs,signal)

	# Step 2: take convolution of the absolute value
	weights = np.repeat(window_height, window_len)
	convolution=np.convolve(weights,signal_abs,'same')
	convolution=convolution/window_len

	# Step 3: find the indices of local maxima of the convolution
	local_max=detect_peaks(convolution, mph=noise_level*5, mpd=window_len/2, show=True)

	# Step 4: locate/save spike vectors
	m=len(local_max)
	n=window_len
	detected_spikes=np.zeros((m,n))
	index=0
	for item in local_max:
		detected_spikes[index]=signal[item-window_len/2:item+window_len/2]
		index=index+1
		
	# Step 5: align spikes 
	k=rand.randint(0,m-1)
	max_location=detected_spikes[k].argmax(axis=0)
	for i in range(0,m-1):
		spike_max_location=detected_spikes[i].argmax(axis=0)
		distance=max_location-spike_max_location
		detected_spikes[i]=np.roll(detected_spikes[i],distance)

	return detected_spikes



















