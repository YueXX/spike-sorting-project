"""Estimates waveforms for a cell over the whole electrode array.
"""
import os.path
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.interpolate import interp1d
from para_fun import param2str

#from matched_filtering import *
from preprocess_data import preprocess_data
# #from waveform_estimation import *
# from cell_estimation import *


DATA_PATH = "/Users/starry1990/Documents/spike_sorting_real_data/data/electrode"
PLOT_PATH = "/Users/starry1990/Documents/spike_sorting_real_data/plots/"
UPSAMPLE_DIR = "/Users/starry1990/Documents/spike_sorting_real_data/upsampled_snippets/"
LOAD_PATH = " /Users/starry1990/Documents/spike_sorting_real_data/clustering/"

# DATA_PATH = "/Users/cfgranda/Google Drive/spike_sorting/data/electrode"
# WAV_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/clustering/"
# LOAD_PATH =  "/Users/cfgranda/Google Drive/spike_sorting_results/wav_est/"
# PLOT_PATH =  "/Users/cfgranda/Google Drive/spike_sorting_results/test/"
# # DATA_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting/data/electrode"
# WAV_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/clustering/"
# LOAD_PATH =  "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/wav_est/"


def waveform_from_spikes_ini_test():
  """Test for waveform_from_spikes_ini. We shift the spikes from an electrode
  and estimate the shift and the waveform, comparing the results to the 
  waveform obtained previously from those spikes.
  """
  colors = "krgmcb"
  electrodes = [58]  
  uf = 5
  wl_down = 41
  h_down = (wl_down - 1)/2
  thresh_load = 0.4
  tol = 1e-3
  thresh = 0.4
  max_iter = 100
  min_spikes = 100
  wl_up_large = 1201
  wl_up = (wl_down - 1) * uf + 1
  shifts = [-400, -50, 60, 150]
  for e in electrodes:
    # We take a spike train corresponding to a certain waveform in an electrode 
    # and check whether the method is capable of finding it
    data_file = DATA_PATH + str(e) + ".npy"
    data_electrode = np.load(data_file)
    data = preprocess_data(data_electrode)
    load_file = (LOAD_PATH + "uf" + str(uf) + "wl" + str(wl_down) 
                + "/electrode" + str(e) + "_thresh_" + param2str(thresh_load) 
                + "_tol_" + param2str(tol) + "_selected.npz")
    # Load results
    auxload = np.load(load_file)
    waveforms = auxload["wav"]
    spikes = auxload["spikes"]
    for shift in shifts:
      aux_spikes = spikes[0] + shift
      aux_spikes = aux_spikes[aux_spikes/uf + h_down < len(data)]
      aux_spikes = aux_spikes[aux_spikes/uf - h_down >= 0]      
      print "Number of spikes: " + str(len(aux_spikes))
      (waveform_est, fit,
       shift_est) = waveform_from_spikes_ini(aux_spikes, data, uf, wl_up_large, 
                                             wl_up, thresh, min_spikes, 
                                             plot_res=True, verbose=True)
      print "Shift: " + str(shift) + " Estimated: " + str(-shift_est)
      fig = plt.figure(figsize=(35.0, 15.0))
      ax = fig.add_subplot(1, 2, 1)
      plt.plot(waveform_est, "--rd")
      plt.plot(waveforms[0], "--bx", label="Original waveform")
      ax.set_title("Shift: " + str(shift) + " Waveform")
      ax = fig.add_subplot(1, 2, 2)
      plt.plot(np.sort(fit), "--o")
      ax.set_title("Fit")
      plot_file = (PLOT_PATH + "cell_est_shift" + str(shift) + ".jpeg")
      fig.savefig(plot_file)
      plt.close(fig)
      
          
def waveform_from_spikes_test_real():
  """Test for waveform_from_spikes. We shift the spikes from an electrode and
  estimate the shift and the waveform, comparing the results to the waveform
  obtained previously from those spikes.
  """
  colors = "krgmcb"
  e_spikes = 58
  electrodes = range(59, 90)
  # [327, 323, 331, 320, 328, 336, 316, 324, 332, 340, 65, 57,
               #  49, 61, 53]
  uf = 5
  wl_down = 41
  h_down = (wl_down - 1)/2
  thresh_load = 0.4
  tol = 1e-3
  thresh = 0.4
  thresh_new = 0.4
  max_iter = 100
  min_spikes = 10
  wl_up_large = 801
  wl_up = (wl_down - 1) * uf + 1
  load_file = (LOAD_PATH + "uf" + str(uf) + "wl" + str(wl_down) 
                + "/electrode" + str(e_spikes) + "_thresh_" + param2str(thresh_load) 
                + "_tol_" + param2str(tol) + "_selected.npz")
  auxload = np.load(load_file)
  waveforms = auxload["wav"]
  spikes = auxload["spikes"]
  for e in electrodes:
    print "Electrode " + str(e)
    data_file = DATA_PATH + str(e) + ".npy"
    data_electrode = np.load(data_file)
    data = preprocess_data(data_electrode)
    # Load results
    (est_res, shift_est, 
     waveform_found) = waveform_from_spikes(spikes[0], data, uf, wl_up_large, 
                                            wl_up, thresh, thresh_new, min_spikes, 
                                            max_iter, tol, plot_res=True, 
                                            verbose=True)
    if waveform_found:                                            
      waveform_est = est_res[-1][0][0]      
      print "Estimated shift: " + str(shift_est)
      fig = plt.figure(figsize=(35.0, 15.0))
      ax = fig.add_subplot(1, 3, 1)
      for i in range(len(est_res)):
        plt.plot(est_res[i][0], label="iter " + str(i))
        plt.legend()
      ax.set_title("Electrode: " + str(e) + " shift: " + str(shift_est) 
                   + " Waveforms")
      ax = fig.add_subplot(1, 3, 2)
      for i in range(1, len(est_res)):
        plt.plot(np.sort(est_res[i][2]), "--o", label="iter " 
                 + str(i))
      plt.legend()
      ax.set_title("Original spikes rnmse")
      ax = fig.add_subplot(1, 3, 3)
      for i in range(1, len(est_res)):
        plt.plot(np.sort(est_res[i][4]), "--o", label="iter " 
                 + str(i))
      plt.legend()
      ax.set_title("New spikes rnmse")
      plot_file = (PLOT_PATH + "cell_est_e" + str(e) + ".jpeg")
      fig.savefig(plot_file)
      plt.close(fig)
    else:
      print "Waveform not found" 


def main():   
  waveform_from_spikes_ini_test()
  # waveform_from_spikes_test_real()


if __name__ == "__main__":
  main()