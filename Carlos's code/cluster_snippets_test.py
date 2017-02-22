"""Test for cluster_snippets.
"""
#from IPython.terminal.embed import InteractiveShellEmbed
import time
import numpy as np
import matplotlib.pyplot as plt
from cluster_snippets import cluster_snippets
from cluster_snippets_plot import plot_clusters
import local_directories as ldir
from preprocess_data import preprocess_data
from segment_data import segment_data
from locate_snippets import locate_snippets
from upsample_snippets import upsample_snippets


def cluster_snippets_test(electrodes):
  """ Test for cluster_snippets.
  """
  wl = 21
  uf = 5
  max_fraction = 0.4
  min_fraction = 0.35
  plt.ion()
  k_val = range(2, 5)
  n_pc_val = [0, 2, 5, 40]
  row_titles = ["k = 2", "k = 3", "k = 4"]
  col_titles = ["No proj", "PC = 2 (av)", "PC = 2",
                "PC = 5 (av)", "PC = 5", "PC = 40 (av)",
                "PC = 40"]
  for e in electrodes:
    data_electrode = np.load(ldir.DATA_PATH + str(e) + ".npy")
    data = preprocess_data(data_electrode)
    ind_ini, ind_end = segment_data(data, wl, max_fraction, min_fraction)
    print "Locating snippets"
    start = time.clock()
    extrema = locate_snippets(data, ind_ini, ind_end, wl)
    end = time.clock()
    print "Locating snippets takes: " + str(end-start)
    print "Upsampling"
    start = time.clock()
    upsamp_res = upsample_snippets(data, extrema, wl, uf)
    end = time.clock()
    print "Upsampling takes: " + str(end-start)
    for wav_type in ["minmax", "maxmin"]:
      snippets = upsamp_res[wav_type + "_snips"]
      locations = upsamp_res[wav_type + "_loc"]
      wav_list = []
      for k in k_val:
        # print "k: " + str(k)
        wav_list_k = []
        for n_pc in n_pc_val:
          # print "n pc: " + str(n_pc)
          if n_pc == 0:
            PCA = False
          else:
            PCA = True 
          av = True
          res = cluster_snippets(snippets, locations, data, k, PCA, n_pc, av)
          n_points = []
          for i in range(len(res["loc"])):
            n_points.append(len(res["loc"][i]))
          wav_list_k.append([res["wav"], n_points])
          if PCA:
            av = False
            res = cluster_snippets(snippets, locations, data, k, PCA, n_pc, av)
            n_points = []
            for i in range(len(res["loc"])):
              n_points.append(len(res["loc"][i]))
            wav_list_k.append([res["wav"], n_points])
        wav_list.append(wav_list_k)
      plot_clusters(wav_list, row_titles, col_titles, save_fig="plot_fig")
      
      # plt.title("Electrode " + str(e))


def main():
  plt.close("all")
  electrodes = [17]
  cluster_snippets_test(electrodes)
  #ipshell = InteractiveShellEmbed()
  # ipshell()


if __name__ == "__main__":
    main()
