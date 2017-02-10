def process_aligned_signal(signal,timeline_list,num_spike,spike_len):
	
	# initialization
	num_electron=signal.shape[0]
	num_cell=len(timeline_list)
	aligned_spike=np.zeros((num_spike,spike_len*num_electron))
	label=[]
	i=0

	for index in range(num_cell):
		spike_loc=timeline_list[index,0:-2]

		for location in spike_loc:
			aligned_spike[i]=signal[:,location:location+spike_len].flatten()
			i=i+1
			label.append(index)

	# choose one row as rolling benchmark
	k=rand.randint(0,num_spike)
	max_location=aligned_spikes[k].argmax(axis=0)

	# roll other rows according to this information
	for i in range(num_spike):
		spike_max_location=alinged_spikes[i].argmax(axis=0)
		distance=max_location-spike_max_location
		aligned_spikes[i]=np.roll(aligned_spikes[i],distance)

	return aligned_spike,label


