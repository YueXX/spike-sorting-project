"""Plots snippets and their 2D projections.

  Plots snippets, their 2D projections and the corresponding
  principal components.

  Args:
    n_electrodes: Number of electrodes.
    data_path: Path to load data.
    plot_path: Path to save results.
    save_singval: Path to save singular values.
"""
import os.path
#from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
import numpy as np
import local_directories as ldir


def PCA_projection(snippets, p):
  aux_mean = np.mean(snippets, axis = 0)
  centered_snips = snippets - aux_mean
  U_mat, singval, singvec = np.linalg.svd(centered_snips,
                                          full_matrices=False)
  pc = singvec[0:p, :]
  points = np.dot(centered_snips, pc.T)
  return pc, points


def plot_upsampled_snippets(uf, wl_down, electrodes):
  """Plots results from upsample_snippets.
  """
  directory = ldir.UPSAMPLE_DIR + "uf" + str(uf) + "wl" + str(wl_down)
  plot_dir = ldir.PLOT_PATH("/usr/local/google/home/cfgranda/Google Drive"
             "/spike_sorting_results/plots/upsampled_snippets/"
             "proj2D/uf" + str(uf) + "wl" + str(wl))
  for e in electrodes:
    print "Electrode " + str(e)
    loadfile = (directory + "/electrode_" + str(e) + ".npz")
    auxload = np.load(loadfile)
    minmax_snips = auxload["minmax_snips"]
    maxmin_snips = auxload["maxmin_snips"]

    plt.figure(figsize=(35.0, 15.0))
    plt.subplot(2, 4, 1)
    for i in range(0, maxmin_snips.shape[0]):
      plt.plot(maxmin_snips[i])
    maxmin_pc, maxmin_proj = PCA_projection(maxmin_snips, 2)
    plt.subplot(2, 4, 2)
    plt.plot(maxmin_pc[0, :])
    plt.xlabel("1st PC")
    plt.subplot(2, 4, 3)
    plt.plot(maxmin_pc[1, :])
    plt.xlabel("2nd PC")
    plt.subplot(2, 4, 4)
    plt.plot(maxmin_proj[:, 0], maxmin_proj[:, 1], "k.")
    plt.xlabel(("Proj"))
    plt.subplot(2, 4, 5)
    for i in range(0, minmax_snips.shape[0]):
      plt.plot(minmax_snips[i])
    minmax_pc, minmax_proj = PCA_projection(minmax_snips, 2)
    plt.subplot(2, 4, 6)
    plt.plot(minmax_pc[0, :])
    plt.xlabel("1st PC")
    plt.subplot(2, 4, 7)
    plt.plot(minmax_pc[1, :])
    plt.xlabel("2nd PC")
    plt.subplot(2, 4, 8)
    plt.plot(minmax_proj[:, 0], minmax_proj[:, 1], "k.")
    plt.xlabel(("Proj"))
    plot_string = (plot_dir + "/electrode" + str(e) + ".jpeg")
    plt.savefig(plot_string)
    plt.close()
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(singval_minmax[0:40], "--x")
    # plt.title("Minmax")
    # plt.subplot(2, 2, 3)
    # plt.semilogy(singval_minmax[0:40], "--x")
    # plt.subplot(2, 2, 2)
    # plt.plot(singval_maxmin[0:40], "--x")
    # plt.title("Maxmin")
    # plt.subplot(2, 2, 4)
    # plt.semilogy(singval_maxmin[0:40], "--x")
    # plot_singval = (save_singval + "/electrode" + str(e) + ".jpeg")
    # plt.savefig(plot_singval)
    # plt.close()
