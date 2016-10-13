"""Script to upsample snippets. 
"""
import os.path
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
"""Script to apply upsample_snippets. 
"""
import numpy as np
from scipy.interpolate import interp1d

from locate_snippets import locate_snippets
from preprocess_data import preprocess_data
from segment_data import segment_data
from upsample_snippets import upsample_snippets
import local_directories as ldir


def main():
  print "Upsampling electrode data"
  wl_down = 21
  uf = 5
  min_fraction = 0.15
  max_fraction = 0.2
  n_electrodes = 512
  directory = ldir.UPSAMPLE_DIR + "uf" + str(uf) + "wl" + str(wl_down)
  if not os.path.exists(directory):
    os.makedirs(directory)
  for e in range(0, n_electrodes):
    print "Electrode " + str(e)
    data_file = ldir.DATA_PATH + str(e) + ".npy"
    data_electrode = np.load(data_file)
    data = preprocess_data(data_electrode)
    ind_ini, ind_end = segment_data(data, wl_down, max_fraction, min_fraction)
    extrema = locate_snippets(data, ind_ini, ind_end, wl_down)
    savepath = directory + "/electrode_" + str(e)
    debug = False
    upsample_snippets(data, extrema, wl_down, uf, savepath, debug)


if __name__ == "__main__":
    main()