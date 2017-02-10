"""
Clustering-based spike sorting for a single electrode
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


def clustersort_params(uf=5, wl_down=21, max_frac=0.20, min_frac=0.15, k = 3,
                       k_axon = 3, k_soma = 4, PCA=False, n_pc=5, av=True):
  """ Initializes parameters for clustersort.
  
  Args:
    uf: Upsampling factor.
    wl_down: Window length for each waveform.
    max_frac: Upper bound on the energy of the snippets (see segment_data).
    min_frac: Lower bound on the energy of the snippets (see segment_data).
    k: Parameter for k means.
    PCA: If True we project onto n_pc principal components before clustering.
    n_pc : Number of principal components.
    av: Determines whether the method returns the low-dimensional k-means 
        (projected onto the original space) (average=False) or averages 
        the snippets in each cluster (average=True).

  Returns:
    params: Dictionary containing parameters for clustersort.
  """
  k = {"soma": k_soma, "axon": k_axon}
  params = {"uf": uf, "wl_down": wl_down, "max_frac": max_frac, 
            "min_frac": min_frac, "k": k, "PCA": PCA, "n_pc": n_pc, "av": av}
  return params

def clustersort(data, p):
  """ Clustering-based spike sorting for a single electrode.
  
  Args:
    data: Data.
    p: Parameters (see clustersort_params above).
    
  Returns:
    res: Dictionary containing:
         -waveforms: List of waveforms.
         -spikes: Locations of the spikes corresponding to each waveform.
    
  """
  start = time.clock()
  ind_ini, ind_end = segment_data(data, p["wl_down"], p["max_frac"], 
                                  p["min_frac"])
  end = time.clock()
  print "Segmenting data: " + str(end-start)
  start = time.clock()
  extrema = locate_snippets(data, ind_ini, ind_end, wl)
  end = time.clock()
  print "Locating snippets: " + str(end-start)
  start = time.clock()
  upsamp_res = upsample_snippets(data, extrema, p["wl_down"], p["uf"])
  end = time.clock()
  print "Upsampling snippets: " + str(end-start)
  res = {"waveforms": [], "spikes": [], "fit": []}
  start = time.clock()
  for wav_type in ["soma", "axon"]:
    snippets = upsamp_res[wav_type + "_snips"]
    locations = upsamp_res[wav_type + "_loc"]
    cluster_res = cluster_snippets(snippets, locations, data, p["k"][wav_type],
                                   p["PCA"], p["n_pc"], p["av"])
    for i in range(0,  p["k"][wav_type]):
      res["waveforms"].append(cluster_res["wav"][i, :])
      res["spikes"].append(cluster_res["loc"][i])
  end = time.clock()
  print "Clustering snippets: " + str(end-start)
  return res