"""
"""
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
import local_directories as ldir
# from IPython.terminal.embed import InteractiveShellEmbed
# from evaluate_fit import indices_fit_plot
from preprocess_data import preprocess_data

# DATA_PATH = "/Users/cfgranda/Google Drive/spike_sorting/data/electrode"
# UPSAMPLE_DIR = "/Users/cfgranda/Google Drive/spike_sorting_results/upsampled_snippets/"
# LOAD_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/clustering/"
# PLOT_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/plots/upsampled_snippets/kmeans_clusters/"
# DATA_FILE = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting/data/electrode"

def plot_clusters(wav_list, row_titles, col_titles, save_fig=""):
  """Plots clusters of waveforms.

  Plots the clusters produced for different values of k in the
  k-means algorithm and for different values of p, where p is
  the number of principal components onto which the snippets are
  projected before applying k mean.

  Args:
    wav_list: Waveforms.
    row_titles: Title for each row.
    col_titles: Title for each column.
    save_fig: File name to save plots.
  """
  fig = plt.figure(figsize=(35.0, 15.0))
  n_rows = len(wav_list)
  n_cols = len(wav_list[1])
  for i_row in range(0, n_rows):
    for i_col in range(0, n_cols):
      if i_row == 0 and i_col == 0:
        ax = fig.add_subplot(n_rows, n_cols, i_col +
                             n_cols * i_row + 1)
        ax1 = ax
      else:
        ax = fig.add_subplot(n_rows, n_cols, i_col +
                             n_cols * i_row + 1, sharey=ax1)
      if i_col == 0:
        ax.set_ylabel(row_titles[i_row])
      if i_row == n_rows - 1:
        ax.set_xlabel(col_titles[i_col])
      wav, n_points = wav_list[i_row][i_col]
      for i_wav in range(0, wav.shape[0]):
        ax.plot(wav[i_wav, :], label=str(n_points[i_wav]))
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=wav.shape[0], mode="expand", borderaxespad=0., fontsize=6)
      plt.show()
      # ax.legend(loc="lower right")
  if save_fig:
    fig.savefig(save_fig)
    #fig.show()
    plt.close(fig)
    
    
def plot_mse(mse_list, row_titles, col_titles,
             save_fig=""):
  """Plots ordered mse for each waveform.

  Plots the mse for each waveform for different values of k in the
  k-means algorithm and for different values of p, where p is
  the number of principal components onto which the snippets are
  projected before applying k means.

  Args:
    mse_list: List containing mse values.
    row_titles: Title for each row.
    col_titles: Title for each column.
    save_fig: File name to save plots.
  """
  fig = plt.figure(figsize=(35.0, 15.0))
  n_rows = len(mse_list)
  n_cols = len(mse_list[1])
  for i_row in range(0, n_rows):
    for i_col in range(0, n_cols):
      if i_row == 0 and i_col == 0:
        ax = fig.add_subplot(n_rows, n_cols, i_col +
                             n_cols * i_row + 1)
        ax1 = ax
      else:
        ax = fig.add_subplot(n_rows, n_cols, i_col +
                             n_cols * i_row + 1, sharey=ax1)
      if i_col == 0:
        ax.set_ylabel(row_titles[i_row])
      if i_row == 0:
        ax.set_title(col_titles[i_col])
      mse, data_ms, n_points = mse_list[i_row][i_col]
      for i_wav in range(0, mse.shape[0]):
        norm_mse = np.sqrt(np.divide(mse[i_wav], data_ms[i_wav]))
        sorted_norm_mse = np.sort(norm_mse)
        ax.plot(sorted_norm_mse, label=str(n_points[i_wav]))
      ax.legend(loc="upper left")
  if save_fig:
    fig.savefig(save_fig)
  plt.close(fig)


def plot_fit(load_file, data, snippets, locations, percentile_val,
             n_examples, row_titles, window_length, upsamp_factor,
             save_fig=""):
  """Plots fit for different snippets.

  Plots fit for snippets that have different values of normalized root
  mse for different values of k in the k-means algorithm and for different
  values of p, where p is the number of principal components onto which the
  snippets are projected before applying k means.

  Args:
    load_file: File to load clustering information.
    data: Electrode data.
    snippets: Upsampled snippets.
    locations: Snippet locations.
    percentile_val: Percentiles to be plotted. Contains values between 0 and
                    1. If there are n snippets, we will plot n_examples
                    snippets that are around the (n * p)th in terms of
                    normalized mse for each value p in the array.
    n_examples: Number of snippets to be plotted for each value in
                percentile_val.
    row_titles: Title for each row.
    window_length: Window length.
    upsamp_factor: Upsampling factor.
    save_fig: File name to save plots.
  """
  fig = plt.figure(figsize=(35.0, 15.0))
  auxload = np.load(load_file)
  waveforms = auxload["wav"]
  assignments = auxload["assignments"]
  mse = auxload["mse"]
  data_ms = auxload["data_ms"]
  k = waveforms.shape[0]
  n_rows = len(percentile_val)
  n_cols = k * n_examples
  for i_wav in range(0, k):
    wav = waveforms[i_wav, :]
    wav_ind = np.nonzero(assignments == i_wav)[0]
    n_points = len(wav_ind)
    norm_mse = np.sqrt(np.divide(mse[i_wav], data_ms[i_wav]))
    aux_sort = np.argsort(norm_mse)
    sort_indices = wav_ind[aux_sort]
    n = len(sort_indices)
    for i_perc in range(0, n_rows):
      perc = percentile_val[i_perc]
      if perc == 1:
        index = n - n_examples
      else:
        index = round(perc * n)
      for i in range(0, n_examples):
        ind = sort_indices[index + i]
        norm_mse_snip = norm_mse[aux_sort[index + i]]
        loc = locations[ind]
        (data_ind, wav_ind, ind_coarse,
         ind_fine) = indices_fit_plot(loc, upsamp_factor, window_length)
        if i_wav == 0 and i_perc == 0 and i == 0:
          ax = fig.add_subplot(n_rows, n_cols, i_perc * n_cols +
                               n_examples * i_wav + i + 1)
          ax1 = ax
        else:
          ax = fig.add_subplot(n_rows, n_cols, i_perc * n_cols +
                               n_examples * i_wav + i + 1, sharey=ax1)
        ax.plot(ind_fine, snippets[ind, :], ".b")
        ax.plot(ind_coarse, data[data_ind.astype(int)], "ok", label="Data")
        ax.plot(ind_fine, wav, ".r")
        ax.plot(ind_coarse, wav[wav_ind.astype(int)], "og", label="Fit")
        if i_wav == 0 and i == 0:
          ax.set_ylabel(row_titles[i_perc])
        if i == 0 and i_perc == 0:
          ax.set_title(str(n_points) + " points")
        ax.text(0, 0, str(norm_mse_snip),
                horizontalalignment="left", verticalalignment="bottom",
                transform=ax.transAxes)
  if save_fig:
    fig.savefig(save_fig)
  plt.close(fig)


def compute_n_points(assignments, k):
  """Auxiliary function to compute the number of points in
  each cluster.

  Computes the number of points in each cluster.

  Args:
    assignments: Cluster assignment.
    k: k parameter in k means.

  Returns:
    n_points: Number of points in each cluster.
  """
  n_points = list()
  for i in range(0, k):
    n_points.append(len(np.nonzero(assignments == i)[0]))
  return n_points


def plot_clusters_array(electrodes, cluster_path, save_directory):
  """Plots clustering results for an array of electrodes.

  Plots the clusters produced for different values of k in the
  k-means algorithm and for different values of p, where p is
  the number of principal components onto which the snippets are
  projected before applying k means.

  Args:
    electrodes: Electrodes.
    cluster_path: Path to directory where clustering results are stored.
    save_directory: Directory to save results.
  """
  k_val = range(2, 5)
  p_val = [0, 2, 5, 40]
  row_titles = ["k = 2", "k = 3", "k = 4"]
  col_titles = ["No proj", "PC = 2 (av)", "PC = 2",
                "PC = 5 (av)", "PC = 5", "PC = 40 (av)",
                "PC = 40"]
  for e in electrodes:
    print("Electrode " + str(e))
    waveforms_minmax = list()
    waveforms_maxmin = list()
    for k in k_val:
      waveforms_minmax_k = list()
      waveforms_maxmin_k = list()
      for p in p_val:
        if p == 0:
          p_string = "/no_proj"
        else:
          p_string = "/proj" + str(p)
        loadfile = (cluster_path + p_string + "/av"
                    + "/k" + str(k) + "/electrode"
                    + str(e) + "_minmax.npz")
        auxload = np.load(loadfile)
        wav_minmax_p = auxload["wav"]
        assignments = auxload["assignments"]
        n_points = compute_n_points(assignments, k)
        waveforms_minmax_k.append([wav_minmax_p, n_points])
        loadfile = (cluster_path + p_string + "/av"
                    + "/k" + str(k) + "/electrode"
                    + str(e) + "_maxmin.npz")
        auxload = np.load(loadfile)
        assignments = auxload["assignments"]
        n_points = compute_n_points(assignments, k)
        wav_maxmin_p = auxload["wav"]
        waveforms_maxmin_k.append([wav_maxmin_p, n_points])
        if p > 0:
          loadfile = (cluster_path + p_string + "/no_av"
                      + "/k" + str(k) + "/electrode"
                      + str(e) + "_minmax.npz")
          auxload = np.load(loadfile)
          wav_minmax_p = auxload["wav"]
          assignments = auxload["assignments"]
          n_points = compute_n_points(assignments, k)
          waveforms_minmax_k.append([wav_minmax_p, n_points])
          loadfile = (cluster_path + p_string + "/no_av"
                      + "/k" + str(k) + "/electrode"
                      + str(e) + "_maxmin.npz")
          auxload = np.load(loadfile)
          wav_maxmin_p = auxload["wav"]
          assignments = auxload["assignments"]
          n_points = compute_n_points(assignments, k)
          waveforms_maxmin_k.append([wav_maxmin_p, n_points])
      waveforms_minmax.append(waveforms_minmax_k)
      waveforms_maxmin.append(waveforms_maxmin_k)
    save_fig = (save_directory + "/electrode_" + str(e)
                + "_minmax.jpeg")
    plot_clusters(waveforms_minmax, row_titles, col_titles,
                  save_fig)
    save_fig = (save_directory + "/electrode_" + str(e)
                + "_maxmin.jpeg")
    plot_clusters(waveforms_maxmin, row_titles, col_titles,
                  save_fig)


def plot_mse_array(electrodes, cluster_path, save_directory):
  """Plots normalized root mse for an array of electrodes.

  Plots normalized root mse for different values of k in the
  k-means algorithm and for different values of p, where p is
  the number of principal components onto which the snippets are
  projected before applying k means.

  Args:
    electrodes: Electrodes.
    cluster_path: Path to directory where clustering results are stored.
    save_directory: Directory to save results.
  """
  k_val = range(2, 5)
  p_val = [0, 2, 5, 40]
  row_titles = ["k = 2", "k = 3", "k = 4"]
  col_titles = ["No proj", "PC = 2", "PC = 5", "PC = 40"]
  for e in electrodes:
    print("Electrode " + str(e))
    mse_minmax = list()
    mse_maxmin = list()
    for k in k_val:
      mse_minmax_k = list()
      mse_maxmin_k = list()
      for p in p_val:
        if p == 0:
          p_string = "/no_proj"
        else:
          p_string = "/proj" + str(p)
        loadfile = (cluster_path + p_string + "/av"
                    + "/k" + str(k) + "/electrode"
                    + str(e) + "_minmax.npz")
        auxload = np.load(loadfile)
        mse_minmax_p = auxload["mse"]
        data_ms = auxload["data_ms"]
        assignments = auxload["assignments"]
        n_points = compute_n_points(assignments, k)
        mse_minmax_k.append([mse_minmax_p, data_ms, n_points])
        loadfile = (cluster_path + p_string + "/av"
                    + "/k" + str(k) + "/electrode"
                    + str(e) + "_maxmin.npz")
        auxload = np.load(loadfile)
        data_ms = auxload["data_ms"]
        assignments = auxload["assignments"]
        n_points = compute_n_points(assignments, k)
        mse_maxmin_p = auxload["mse"]
        mse_maxmin_k.append([mse_maxmin_p, data_ms, n_points])
      mse_minmax.append(mse_minmax_k)
      mse_maxmin.append(mse_maxmin_k)
    save_fig = (save_directory + "/electrode_" + str(e)
                + "_minmax.jpeg")
    plot_mse(mse_minmax, row_titles, col_titles,
             save_fig)
    save_fig = (save_directory + "/electrode_" + str(e)
                + "_maxmin.jpeg")
    plot_mse(mse_maxmin, row_titles, col_titles,
             save_fig)


def plot_fit_array(electrodes, cluster_path, snippets_path, data_path,
                   window_length, upsamp_factor, save_directory):
  """Plots fit for an array of electrodes.

  Plots fit for snippets that have different values of normalized root
  mse for different values of k in the k-means algorithm and for different
  values of p, where p is the number of principal components onto which
  the snippets are projected before applying k means.

  Args:
    electrodes: Electrodes.
    cluster_path: Path to directory where clustering results are stored.
    snippets_path: Path to load snippets.
    data_path: Path to load electrode data.
    window_length: Window length.
    upsamp_factor: Upsampling factor.
    save_directory: Directory to save results.
  """
  k_val = range(2, 5)
  p_val = [0, 2, 5, 40]
  row_titles = ["Best", "25%", "50%", "75%", "Worst"]
  n_examples = 3
  percentile_val = [0., 0.25, 0.5, 0.75, 1.]
  # col_titles = ["No proj", "PC = 2", "PC = 5", "PC = 40"]
  for e in electrodes:
    print("Electrode " + str(e))
    data_file = data_path + str(e) + ".npy"
    data_electrode = np.load(data_file)
    data = preprocess_data(data_electrode)
    load_file_snippets = (snippets_path + "/electrode_" + str(e)
                          + ".npz")
    auxload = np.load(load_file_snippets)
    minmax_snips = auxload["minmax_snips"]
    maxmin_snips = auxload["maxmin_snips"]
    minmax_loc = auxload["minmax_loc"]
    maxmin_loc = auxload["maxmin_loc"]
    for k in k_val:
      for p in p_val:
        if p == 0:
          p_string = "/no_proj"
        else:
          p_string = "/proj" + str(p)
        load_file = (cluster_path + p_string + "/av"
                     + "/k" + str(k) + "/electrode"
                     + str(e) + "_minmax.npz")
        save_fig = (save_directory + p_string + "/k" + str(k)
                    + "/electrode_" + str(e) + "_minmax.jpeg")
        plot_fit(load_file, data, minmax_snips, minmax_loc,
                 percentile_val, n_examples, row_titles,
                 window_length, upsamp_factor, save_fig)
        load_file = (cluster_path + p_string + "/av"
                     + "/k" + str(k) + "/electrode"
                     + str(e) + "_maxmin.npz")
        save_fig = (save_directory + p_string + "/k" + str(k)
                    + "/electrode_" + str(e) + "_maxmin.jpeg")
        plot_fit(load_file, data, maxmin_snips, maxmin_loc,
                 percentile_val, n_examples, row_titles,
                 window_length, upsamp_factor, save_fig)


def main():
  electrodes = range(512)
  window_length = 21
  upsamp_factor = 5
  snippets_path = (ldir.UPSAMPLE_DIR + "uf" + str(upsamp_factor) + "wl" 
                   + str(window_length))
  load_directory = (ldir.LOAD_PATH+ "uf" + str(upsamp_factor)+ "wl"
                    + str(window_length))
  plot_directory = (ldir.PLOT_PATH + "uf" + str(upsamp_factor)
                    + "wl" + str(window_length) + "/waveforms")
  if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
  plot_clusters_array(electrodes, load_directory, plot_directory)
  plot_directory_mse = (ldir.PLOT_PATH + "uf" + str(upsamp_factor)
                        + "wl" + str(window_length) + "/mse")
  if not os.path.exists(plot_directory_mse):
    os.makedirs(plot_directory_mse)
  plot_mse_array(electrodes, load_directory, plot_directory_mse)
  plot_directory_fit = (ldir.PLOT_PATH + "uf" + str(upsamp_factor)
                        + "wl" + str(window_length) + "/fit")
  if not os.path.exists(plot_directory_fit):
    os.makedirs(plot_directory_fit)
  plot_fit_array(electrodes, load_directory, snippets_path, DATA_FILE,
                 window_length, upsamp_factor, plot_directory_fit)
  # ipshell = InteractiveShellEmbed()
  # ipshell()

if __name__ == "__main__":
  main()
