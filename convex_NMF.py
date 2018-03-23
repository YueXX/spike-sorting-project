from snmf import *
import numpy as np
from semi_nmf_base import *
from data_generating import *
from NMF_functions import *
from cnmf import *
from data_conv import *

# generating signal
num_neuron = 5
neuron_firing_rate_parm1 = np.arange(10, 10 + 5* num_neuron, 5)

num_electron = 40
time_l = 100
time_step = 1

neuron_shape_parm1 = neuron_shape_generator(num_neuron)
undersample_window = len(np.arange(-np.pi, np.pi, time_step))
undersample_threshold = 0.9
convolution_window = int(1.4 * undersample_window)

electron_indicator_matrix1 = neuron_func_connect_matrix_generator(num_electron, num_neuron, 0.1)
# print(electron_indicator_matrix1)
# print('rank', np.linalg.matrix_rank(electron_indicator_matrix1))
signal1, neuron_spiking_timeline1, signal_matrix = multiple_electron_generator(time_step, time_l, neuron_firing_rate_parm1,
                                                                neuron_shape_parm1, electron_indicator_matrix1)

undersample_signal1,convolution_signal1, spike_loc = undersample_signal(signal1, convolution_kernel2, convolution_window, undersample_window, undersample_threshold)
undersample_signal_basis, convolution_signal_basis, spike_loc_basis = undersample_signal(signal_matrix,
                                                                                         convolution_kernel2,
                                                                                         convolution_window,
                                                                                         undersample_window,
                                                                                         undersample_threshold)

png_name = 'fig/rank_nmf'
iterations = 5000
# rank_list = np.arange(5,30)
# error, W_list, H_list = semi_NMF_vs_rank(undersample_signal1.transpose(), rank_list, iterations, png_name)

rank_list = np.array([num_neuron])
#error, W_list, H_list = semi_NMF_vs_rank(undersample_signal1.transpose(), rank_list, iterations, png_name)

# try fake data
fake_data = np.dot(electron_indicator_matrix1, undersample_signal_basis).transpose()

rank = num_neuron
cnmf_mdl = CNMF(fake_data, num_bases=rank)
cnmf_mdl.factorize(niter=iterations)
H = cnmf_mdl.H

error_undersample = np.linalg.norm(undersample_signal1 - np.dot(electron_indicator_matrix1, undersample_signal_basis)) / np.linalg.norm(undersample_signal1)
print('error undersample', error_undersample)
print('error of base', np.linalg.norm(electron_indicator_matrix1.transpose()-H)/np.linalg.norm(electron_indicator_matrix1))

print(np.round(H,2))
