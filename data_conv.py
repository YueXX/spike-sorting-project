import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_generating import *


def convolution_kernel(data, h):
    windowed_data_sq = np.convolve(data ** 2, np.ones(h + 1), "same")
    windowed_data_aux = np.sqrt(windowed_data_sq)
    windowed_data = windowed_data_aux / np.amax(np.abs(windowed_data_aux))

    return windowed_data


def convolution_kernel2(data, h):
    """
    :param data:
    :param h: must be even
    :return:
    """
    window_l = np.hstack((np.ones(int(h/2)), -np.ones(int(h/2))))
    windowed_data_l = np.convolve(data, window_l, "same")
    #windowed_data_l = windowed_data_l/np.max(np.abs(windowed_data_l))

    return windowed_data_l


def convolution_signal_func(signal_matrix, convolution_kernal, convolution_window, normalize=False):
    num_row = signal_matrix.shape[0]
    convolution_signal = np.zeros((num_row, signal_matrix.shape[1]))
    for i in range(num_row):
        convolution_signal[i] = convolution_kernal(signal_matrix[i], convolution_window)
    if normalize:
        convolution_signal = convolution_signal/np.max(np.abs(convolution_signal))
    return convolution_signal


def undersample(convolution_data, undersample_window, undersample_threshold):
    """
    undersample convoluved signal
    ======================================================
    mat,loc_int=undersample(conv,window_l,threshold)
    input:
    conv: convolved signal matrix
    window_l: window length for undersampling
    threshold: the threshold for undersampling
    output:
    mat:the signal matrix after convolution, max of mat
    is 1, min is 0 or -1 depending on the kernel
    loc_int: the location of where the matrix has non zero
    element, where there's a spike
    ======================================================
    """
    # input value convolved signals, window length and threshold for undersample
    interval_num = int(convolution_data.shape[0] / undersample_window)
    mat = np.zeros(interval_num)
    loc_list = []
    for i in range(interval_num):
        undersample_interval = convolution_data[undersample_window * i:undersample_window * (i + 1)]
        interval_abs = np.abs(undersample_interval)
        indices = [x for x in interval_abs if x >= undersample_threshold]
        if len(indices) == 0:
            mat[i] = 0
        else:
            loc = undersample_window * i + np.argmax(interval_abs)
            mat[i] = convolution_data[int(loc)]
            loc_list.append(int(loc))
    return mat, np.array(loc_list)


def undersample_signal_func(convolution_matrix, undersample_window, undersample_threshold):
    """
    # perform convolution and then undersample to signal matrix

    :return:
    undersample_signal_mat: undersampling signal matrix
    spike_loc_list: the point where we get undersampling point

    """
    num_electron = convolution_matrix.shape[0]
    signal_length = convolution_matrix.shape[1]
    undersample_signal_length = int(signal_length/undersample_window)
    undersample_signal_mat = np.zeros((num_electron, undersample_signal_length))

    spike_loc_list = []

    for i in range(num_electron):
        convolution_signal = convolution_matrix[i]

        # perform undersample on ith electron
        undersample_signal, loc = undersample(convolution_signal, undersample_window, undersample_threshold)
        undersample_signal_mat[i] = undersample_signal

        # get the location of the signal where we get the max convolution point
        spike_loc_list.append(loc)

    return undersample_signal_mat, spike_loc_list


def undersample_convolution_plot(signal_mat, spike_loc_list, convolution_signal_mat, undersample_window, plot_interval, time_step,
                                 path ='fig/undersample plot'):
    time_axis = np.arange(plot_interval[0], plot_interval[1], time_step)
    num_electron = signal_mat.shape[0]

    fig, axs = plt.subplots(nrows=num_electron, ncols=1, sharex=True, sharey=True)
    # normalize signal for better visualization
    signal_mat = signal_mat/np.max(np.abs(signal_mat))
    convolution_signal_mat = convolution_signal_mat/np.max(np.abs(convolution_signal_mat))
    for i in range(num_electron):
        # plot original signal
        start = int(plot_interval[0]/time_step)
        end = int(plot_interval[1]/time_step)
        axs[i].plot(time_axis, signal_mat[i][start:end], label='signal ')

        # plot convolution
        axs[i].plot(time_axis, convolution_signal_mat[i][start: end], label='convolution')

        # plot the point where we think is the spike location
        spike_loc = spike_loc_list[i]
        for index in spike_loc:

            if index * time_step < plot_interval[1] and index * time_step > plot_interval[0]:
                axs[i].scatter(index * time_step, 1, marker='o', c='r')
        # plot the undersample grid
        grid_point = np.arange(start, end, undersample_window)
        axs[i].scatter(grid_point * time_step, np.zeros(len(grid_point)), marker="|", c= 'g')

        axs[i].set_title('Undersample of electron ' + str(i))

    plt.legend()
    plt.savefig(path)
    plt.close()
    return


def undersample_convolution_heatmap_plot(undersample_mat, plot_length, time_step):
    t = np.arange(0, plot_length, time_step)
    undersample_mat = undersample_mat[:, : len(t)]
    ax = sns.heatmap(undersample_mat)
    plt.savefig('fig/undersample heatmap')
    plt.close()

    return


def undersample_plot(undersample_mat, plot_length, file_name = 'fig/undersample mat'):
    num_neuron = undersample_mat.shape[0]
    fig, axs = plt.subplots(nrows=num_neuron, ncols=1, sharex=True, sharey=True)
    for i in range(num_neuron):
        axs[i].plot(undersample_mat[i][: plot_length])
        axs[i].set_title('undersample ' + str(i) + ' signals')
    plt.savefig(file_name)
    return

