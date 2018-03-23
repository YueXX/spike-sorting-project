# data generated functions built according to Performance evaluation of PCA-based spike sorting algorithms.pdf
import numpy as np
import matplotlib.pyplot as plt


def spike_shape_func(t, amplitude, total_duration, rising_phase, t0):
    spike = amplitude*np.sin((t-t0)/rising_phase)*np.exp(-np.power(t/total_duration, 2))
    return spike


def time_occurrence(time_length, firing_rate):
    # time occurrences during time t, modified by poisson process

    # randomize initial firing time to avoid overlap
    firing_time = np.random.uniform(0, firing_rate)

    time_occurrence_array = []
    while firing_time < time_length:
        new_firing_time = firing_time + np.random.poisson(firing_rate)
        time_occurrence_array.append(firing_time)
        firing_time = int(new_firing_time)
    return np.array(time_occurrence_array[1:])


def gaussian_noise_generator(noise_mean, noise_deviation, time_length):
    noise = np.random.normal(noise_mean, noise_deviation, time_length)
    return noise


def neuron_shape_plot(neuron_shape_parm, time_step, path = 'fig/neuron shape plot'):
    t = np.arange(-np.pi, np.pi, time_step)
    num_neuron = len(neuron_shape_parm['t0'])
    for i in range(num_neuron):
        shape = spike_shape_func(t, neuron_shape_parm['amplitude'][i], neuron_shape_parm['total_duration'][i],
                                 neuron_shape_parm['rising_phase'][i], neuron_shape_parm['t0'][i] )
        plt.plot(t,shape, label= str(neuron_shape_parm['amplitude'][i])+' '+str(neuron_shape_parm['total_duration'][i]) +
                               ' '+str(neuron_shape_parm['rising_phase'][i]) + ' '
                                 + str(neuron_shape_parm['t0'][i]))

    plt.title('neurons shape plot')
    plt.legend()
    plt.savefig(path)
    plt.close()

def single_electron_generator(time_step, time_length, neuron_firing_rate_parm, neuron_shape_parm):
    """
    :param time_step:
    :param time_length:
    :param neuron_firing_rate_parm:
    :param neuron_shape_parm:
    :return:
    """
    num_neuron = len(neuron_shape_parm['amplitude'])
    neuron_spiking_timeline = []
    spike_interval = np.arange(-np.pi, np.pi, time_step)
    signal = np.zeros(int(time_length/time_step))

    for i in range(num_neuron):
        firing_rate = 1.0 * neuron_firing_rate_parm[i] / time_step
        spike_timeline = time_occurrence(int(time_length/time_step)-len(spike_interval), firing_rate)
        neuron_spiking_timeline.append(spike_timeline)
        signal_single_neuron = np.zeros(int(time_length/time_step))
        spike = spike_shape_func(spike_interval, neuron_shape_parm['ampliltude'][i], neuron_shape_parm['total_duration'][i],
                                 neuron_shape_parm['rising_phase'][i], neuron_shape_parm['t0'][i])
        for spike_time in spike_timeline:
            signal_single_neuron[spike_time: spike_time + len(spike_interval)] = spike
        signal += signal_single_neuron

    return signal, neuron_spiking_timeline


def multiple_electron_generator(time_step, time_length, neuron_firing_rate_parm, neuron_shape_parm,
                                electron_indictor_matrix):

    """
    :param time_step: interpolation
    :param time_length: total length
    :param neuron_firing_rate_parm: the firing rate parameters for all neuron
    :param neuron_shape_parm: shape parameters for all neurons
    :param electron_indictor_matrix: each row indicate one electron
    :return:

    """
    num_neuron = len(neuron_shape_parm['amplitude'])
    neuron_spiking_timeline = []
    spike_interval = np.arange(-np.pi, np.pi, time_step)
    signal_length = int(time_length / time_step)
    signal_basis_matrix = np.zeros((num_neuron, signal_length))
    for i in range(num_neuron):
        firing_rate = 1.0 * neuron_firing_rate_parm[i] / time_step
        spike_timeline = time_occurrence(int(time_length / time_step) - len(spike_interval), firing_rate)
        neuron_spiking_timeline.append(spike_timeline)
        signal_single_neuron = np.zeros(int(time_length / time_step))
        spike = spike_shape_func(spike_interval, neuron_shape_parm['amplitude'][i], neuron_shape_parm['total_duration'][i],
                                 neuron_shape_parm['rising_phase'][i], neuron_shape_parm['t0'][i])
        for spike_time in spike_timeline:
            signal_single_neuron[spike_time: spike_time + len(spike_interval)] = spike
        signal_basis_matrix[i] = signal_single_neuron

    signal_per_electron = np.dot(electron_indictor_matrix, signal_basis_matrix)

    return signal_per_electron, neuron_spiking_timeline, signal_basis_matrix


def multiple_electron_plot(signal_per_electron, time_step, plot_time_interval, path='fig/multiple electron plot'):
    num_electron = signal_per_electron.shape[0]
    fig, axs = plt.subplots(nrows=num_electron, ncols=1, sharex=True, sharey=True )
    time_axis = np.arange(plot_time_interval[0], plot_time_interval[1], time_step)
    for i in range(num_electron):
        axs[i].plot(time_axis, signal_per_electron[i][int(plot_time_interval[0]/time_step): int(plot_time_interval[1]/time_step)])
        axs[i].set_title('Electron '+str(i)+' detected signals')
    plt.xlabel('time')
    plt.savefig(path)
    plt.close()
    return


def neuron_func_connect_matrix_generator(num_electron, num_neuron, p):
    """
    :param num_electron:
    :param num_neuron:
    :param p: the probability that a neuron can be detected by an electron
    :return: the indicator matrix
    """
    k = np.random.binomial(1, p, num_electron * num_neuron)
    electron_indicator_matrix = np.reshape(k, (-1, num_neuron))
    return electron_indicator_matrix


def neuron_shape_generator(num_neuron, amplitude_range=[5,10], total_duration_range=[0.5,1.5],
                           rising_phase_range=[-1.5, 1,5], t0_range=[-0.5, 0.5]):
    amplitude_random = np.random.uniform(amplitude_range[0], amplitude_range[1], num_neuron)
    total_duration_random = np.random.uniform(total_duration_range[0], total_duration_range[1], num_neuron)

    # absolute value of rising phase smaller than 0.2 will cause huge oscillation
    rising_phase_random = np.random.uniform(rising_phase_range[0]+0.2, rising_phase_range[1]-0.2, num_neuron)
    for i in range(num_neuron):
        if rising_phase_random[i] < 0:
            rising_phase_random[i] -= 0.2
        else:
            rising_phase_random[i] += 0.2

    t0_random = np.random.uniform(t0_range[0], t0_range[1], num_neuron)

    neuron_shape_parm = {'amplitude': np.round(amplitude_random, 2), 'total_duration': np.round(total_duration_random, 2),
                         'rising_phase': np.round(rising_phase_random, 1), 't0': np.round(t0_random, 2)}

    return neuron_shape_parm


def firing_rate_generator(num_neuron, firing_rate_range=[800, 1000], overlap_interval=10):

    neuron_firing_rate_parm = np.zeros(num_neuron)
    firing_rate = firing_rate_range[0]

    for i in range(num_neuron):
        firing_rate = np.random.uniform(firing_rate + overlap_interval, firing_rate_range[1])
        neuron_firing_rate_parm[i] = firing_rate
    return neuron_firing_rate_parm


# TODO
# Non Gaussian dynamical e.g Ornstein–Uhlenbeck noise generator
# neuron shape generator
# Down-sample signal

########################################################################################################################
# test data generating functions


if __name__ == "__main__":
    # neuron_firing_rate_parm1=[30,50,80]
    # neuron_shape_parm1 = {'amplitude': [10,8,10], 'total_duration': [0.5,1,1], 'rising_phase': [1.5,-0.5,0.5],
    # 't0': [-0.5,0,0.47]}
    # electron_indicator_matrix1 = np.array([[1,0,0], [1,0,0], [1,1,1], [1,1,1]])

    num_neuron = 3
    num_electron = 10
    time_s = 0.05
    time_l = 10000
    neuron_firing_rate_parm1 = firing_rate_generator(num_neuron)
    neuron_shape_parm1 = neuron_shape_generator(num_neuron)
    print(neuron_shape_parm1)
    neuron_shape_plot(neuron_shape_parm1, time_s)
    electron_indicator_matrix1 = neuron_func_connect_matrix_generator(num_electron, num_neuron, 0.7)
    print(electron_indicator_matrix1)
    signal1, neuron_spiking_timeline1, signal_matrix = multiple_electron_generator(time_s, time_l, neuron_firing_rate_parm1,
                                                              neuron_shape_parm1, electron_indicator_matrix1)
    plot_time_length1 = 1000
    multiple_electron_plot(time_s, plot_time_length1, signal1)
