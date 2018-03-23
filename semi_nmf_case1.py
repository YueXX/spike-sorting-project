
from data_generating import *
from data_conv import *
from NMF_functions import *
from collections import Counter
if __name__ == "__main__":
    # set signal parameters
    num_neuron = 20
    num_electron = 100
    neuron_shape_parm = neuron_shape_generator(num_neuron)
    # neuron_indicator_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    neuron_indicator_matrix = neuron_func_connect_matrix_generator(num_electron, num_neuron, p = 0.2)
    print(np.linalg.matrix_rank(neuron_indicator_matrix))
    neuron_firing_rate_parm = firing_rate_generator(num_neuron)

    time_length = 20000
    time_step = 0.05
    # generating signals
    signal, neuron_spiking_timeline, signal_basis_matrix = multiple_electron_generator(time_step, time_length,
                                                                                       neuron_firing_rate_parm,
                                                                                       neuron_shape_parm,
                                                                                       neuron_indicator_matrix)
    # plot signals
    plot_time_interval = [0,10000]
    neuron_shape_plot(neuron_shape_parm, time_step)
    # multiple_electron_plot(signal, time_step, plot_time_interval)

    ###################################################################################################################
    # set convolution & undersample parameters
    undersample_window = int(len(np.arange(-np.pi, np.pi, time_step)))
    undersample_threshold = 0.8
    convolution_window = int(undersample_window)

    # apply convolution to signal
    convolution_matrix = convolution_signal_func(signal, convolution_kernel2, convolution_window)
    # plot convolution
    # multiple_electron_plot(convolution_matrix, time_step, plot_time_interval, path='fig/convolution_plot')

    # apply undersample to convolution signal
    undersample_signal_mat, spike_loc_list = undersample_signal_func(convolution_matrix, undersample_window, undersample_threshold)
    # visualize convolution and undersample
    plot_time_interval = [1300,1325]
    undersample_convolution_plot(signal[0:2, :], spike_loc_list, convolution_matrix, undersample_window, plot_time_interval, time_step, path='fig/undersample plot')

    # check convolution/undersample error: the convolution & undersample of the signal should be approximately the
    # dot product of neuron_indicator_matrix and the convolution & undersample of signal_basis_matrix

    convolution_basis_matrix = convolution_signal_func(signal_basis_matrix, convolution_kernel2, convolution_window)
    undersample_basis_matrix, basis_spike_loc = undersample_signal_func(convolution_basis_matrix, undersample_window, undersample_threshold)

    undersample_error = np.linalg.norm(undersample_signal_mat - np.dot(neuron_indicator_matrix, undersample_basis_matrix))/np.linalg.norm(undersample_signal_mat)
    print('the error cause by undersampling & convolution is:', undersample_error)

    ###################################################################################################################
    # apply NMF to the undersample signal matrix
    rank_list = np.arange(10,30)
    iterations = 1000
    data = undersample_signal_mat.transpose()
    error, W_list, H_list = semi_NMF_vs_rank(data, rank_list, iterations, png_name='fig/semi_NMF_rank')

