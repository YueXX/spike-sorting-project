import numpy as np
import matplotlib.pyplot as plt
from snmf import *

def semi_NMF_vs_rank(matrix, rank_list, iterations, png_name=None):
    error = []
    W_list = []
    H_list = []

    for num_base in rank_list:
        print('finish rank', num_base)
        snmf_mdl = SNMF(matrix, num_bases=num_base)
        snmf_mdl.factorize(niter=iterations)
        W = snmf_mdl.W
        H = snmf_mdl.H
        error_norm = np.linalg.norm(matrix - np.dot(W,H))
        error_norm = error_norm / np.linalg.norm(matrix)
        error.append(error_norm)
        W_list.append(snmf_mdl.W)
        H_list.append(snmf_mdl.H)

    if png_name:
        title = 'Semi Nonngegative matrix factorization' + png_name
        save_name =  str(png_name)

        plt.plot(rank_list, error)
        plt.title(title)
        plt.xlabel('rank')
        plt.ylabel('Semi NMF approximation error')
        plt.savefig(save_name)
        plt.close()
    return error, W_list, H_list


def NMF_basis_plot(W, file_name ='fig/NMF basis plot'):
    # W = W.transpose()
    num_neuron = W.shape[0]
    fig, axs = plt.subplots(nrows=num_neuron, ncols=1, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(num_neuron):
        axs[i].plot(W[i])
    plt.xlabel('time step')
    axs[0].set_title('Semi NMF Basis Matrix F Column Plot')
    plt.savefig(file_name)
    plt.close()
    return

