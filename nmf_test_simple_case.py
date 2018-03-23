from snmf import *
import numpy as np
from semi_nmf_base import *
from data_generating import *
from NMF_functions import *
from data_conv import *

# generating signal
num_neuron = 2
# neuron_firing_rate_parm1 = np.arange(710, 710 + 50* num_neuron, 50)
# neuron_firing_rate_parm1 = firing_rate_generator(num_neuron)
num_electron = 2
time_l = 10000
time_step = 0.05
# neuron_shape_parm1 = neuron_shape_generator(num_neuron)

neuron_firing_rate_parm1 = np.array([200, 230])
electron_indicator_matrix1 = np.eye(2)
neuron_shape_parm1 = {'amplitude': [15, 5], 'total_duration': [1, 1], 'rising_phase': [1.5, -0.5],
                      't0': [-0.5, 0]}

undersample_window = int(len(np.arange(-np.pi, np.pi, time_step)))
undersample_threshold = 0.9
convolution_window = int(undersample_window)

#electron_indicator_matrix1 = neuron_func_connect_matrix_generator(num_electron, num_neuron, 0.5)
signal1, neuron_spiking_timeline1, signal_matrix = multiple_electron_generator(time_step, time_l, neuron_firing_rate_parm1,
                                                                neuron_shape_parm1, electron_indicator_matrix1)

plot_time_length = 10000
# undersample signal
undersample_signal1,convolution_signal1, spike_loc = undersample_signal(signal1, convolution_kernel2, convolution_window, undersample_window, undersample_threshold)

# undersample signal basis
undersample_signal_basis, convolution_signal_basis, spike_loc_basis = undersample_signal(signal_matrix,
                                                                                         convolution_kernel2,
                                                                                         convolution_window,
                                                                                         undersample_window,
                                                                                         undersample_threshold)

# plot
neuron_shape_plot(neuron_shape_parm1,time_step)
multiple_electron_plot(time_step, plot_time_length, signal1)

# perform semi NMF algorithm
matrix = undersample_signal1.transpose()
rank_list = np.array([num_neuron])
iterations = 5000

error, W_list, H_list = semi_NMF_vs_rank(matrix, rank_list, iterations)

print('coefficient matrix', H_list[0].transpose())
print('indicator matrix', electron_indicator_matrix1)

NMF_basis_plot(W_list[0])

NMF_basis_plot(undersample_signal_basis.transpose(), file_name= 'fig/true basis')