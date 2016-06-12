import numpy as np
import random as rand
from matplotlib import pyplot as plt
from detect_peaks import detect_peaks

#########################################################
# The following code apply K-means algorithms to signal


# Question 1: we have to manually set the window length/convolution window,  
# since we use convolution window to find spike, the convolution result may not give
# us the best estimate

# Question 2: many spikee detection methods, we choose convolution is to ..

def detect_spike(signal, window_len, noise_level, window_height=2):
	
	################################################
	# Step 1: take the absolute value of signal
	signal=map(abs,signal)

	# Step 2: take convolution of the absolute value
	weights = np.repeat(window_height, window_len)
	convolution=np.convolve(weights,signal,'same')
	
	# Step 3: find local maxima of the convolution
	local_max=detect_peaks(convolution, mph=noise_level*2, mpd=window_len/2, show=True)









