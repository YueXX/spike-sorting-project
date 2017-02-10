"""Test for cluster_snippets.
"""
from IPython.terminal.embed import InteractiveShellEmbed
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


def clustersort_test(electrodes):
  """ Test for clustersort.
  """
  wl = 21
  uf = 5
  max_fraction = 0.4
  min_fraction = 0.35
  plt.ion()
  k_val = range(2, 5)
  n_pc_val = [0, 2, 5, 40]
  for e in electrodes:
    data_electrode = np.load(ldir.DATA_PATH + str(e) + ".npy")
    data = preprocess_data(data_electrode)
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
        k_dic = {"soma": k, "axon": k - 1}
        params = {"uf": uf, "wl_down": wl_down, "max_frac": max_frac, 
                  "min_frac": min_frac, "k": k_dic, "PCA": PCA, "n_pc": n_pc, 
                  "av": av}
        res = clustersort(data, p)
        

def main():
  plt.close("all")
  electrodes = [46]
  cluster_snippets_test(electrodes)
  # ipshell = InteractiveShellEmbed()
  # ipshell()


if __name__ == "__main__":
    main()