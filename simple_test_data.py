from data_generating import *
from data_conv import *
from sklearn.decomposition import NMF
from snmf import *
from semi_nmf_base import *



num_neuron = 5
num_electron = 20
time_s = 0.1
time_l = 1000
neuron_firing_rate_parm1 = firing_rate_generator(num_neuron)
neuron_shape_parm1 = neuron_shape_generator(num_neuron)
neuron_shape_plot(neuron_shape_parm1, time_s)
electron_indicator_matrix1 = neuron_func_connect_matrix_generator(num_electron, num_neuron, 0.1)
#print(electron_indicator_matrix1)
signal1, neuron_spiking_timeline1 = multiple_electron_generator(time_s, time_l, neuron_firing_rate_parm1,
                                                                neuron_shape_parm1, electron_indicator_matrix1)
plot_time_length1 = 1000
#multiple_electron_plot(time_s, plot_time_length1, signal1)

undersample_threshold = 0.6
plot_length = 200
convolution_window = len(np.arange(-np.pi, np.pi, time_s))
undersample_signal_mat = undersample_signal(signal1, convolution_kernel2, convolution_window, convolution_window,
                                            undersample_threshold)
#undersample_convolution_plot(signal1, undersample_signal_mat, convolution_window, plot_length, time_s)

print(np.linalg.matrix_rank(electron_indicator_matrix1))
print(np.linalg.matrix_rank(undersample_signal_mat))


print(np.min(undersample_signal_mat))