"""Estimates waveforms for a cell over the whole electrode array.
"""
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.interpolate import interp1d
from matched_filtering import *
from preprocess_data import preprocess_data
from waveform_estimation import *
from evaluate_fit import evaluate_fit
DATA_PATH = "/Users/cfgranda/Google Drive/spike_sorting/data/electrode"
WAV_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/clustering/"
PLOT_PATH =  "/Users/cfgranda/Google Drive/spike_sorting_results/plots/cell_ini"
LOAD_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/wav_est/"
SAVE_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/cell_est/ini/"
# DATA_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting/data/electrode"
# WAV_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/clustering/"
# SAVE_PATH =  "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/wav_est/"


def cell_ini(spikes, data, uf, wl_up_large, wl_up, thresh, thresh_wav,
             min_spikes, min_length, interp=True, plot_res=False, 
             verbose=False):
  """Estimates waveform from spike locations.

  Averages snippets around the spikes and centers according to the arg minimum 
  of the amplitude.

  Args:
    spikes: Spikes.
    data: Data.
    uf: Upsampling factor.
    wl_up_large: Window for the first average.
    wl_up: Window length of waveform estimate.
    thresh: Threshold for the normalized fit.
    min_spikes: Minimum number of spikes over which we declare that a waveform
                was detected.
    interp: If True, snippets are interpolated before averaging.
    plot_res: If True results are plotted.
    verbose: If True information is printed.
  Returns:
    waveform: Estimated waveform or None if none is found.
    shift: Shift of the waveform center from the spike location.
    fit: Normalized fits.
  """
  # Initialize by averaging upsampled snippets over a large window
  start = time.clock()
  if interp:
      av_snippets = average_waveforms_interp_fast(spikes, data, params["uf"], 
                                                  params["wl_cell_ini"])
  else:
      av_snippets, n_shifts = average_waveforms(spikes, data, params["uf"], params["wl_cell_ini"])
  end = time.clock()
  if verbose:
    print ("Initial average takes " + str(end-start))    
  if plot_res:
    fig = plt.figure(figsize=(35.0, 15.0))
    plt.plot(av_snippets)
    plt.title("Snippet average")
    plot_file = (PLOT_PATH + "av_snippets.jpeg")
    fig.savefig(plot_file)
    plt.close(fig)
  std = std_waveforms(av_snippets, spikes, data, params["uf"])
  err_rel = std / abs(av_snippets)
  waveform, shift = trim_waveform(av_snippets, err_rel, params["uf"], params["thresh_wav"])
  print "Shift: " + str(shift)
  if len(waveform) > min_length:                                  
    if plot_res:
      fig = plt.figure(figsize=(35.0, 15.0))
      plt.plot(waveform)
      plt.title("Snippet average")
      plot_file = (PLOT_PATH + "wav" +".jpeg")
      fig.savefig(plot_file)
      plt.close(fig)
      # ipshell = InteractiveShellEmbed()
      # ipshell()
  else:
    waveform = None
    spikes_thresh_fit = None
    shift = None
    print "Waveform not found"
  return waveform, shift, av_snippets
  
  
def cell_ini_array(e_list_1, e_list_2, uf, wl_up_large, wl_up, thresh, 
                   thresh_wav, min_spikes, min_length, interp, data_path, load_path, 
                   save_path, verbose=False):
  """Applies waveform_from_spikes_ini over the whole array.

  Args:
    e_list_1: List of electrodes whose spikes we use.
    e_list_2: List of electrodes at which we look for waveforms.
    uf: Upsampling factor.
    wl_up_large: Window for the first average.
    wl_up: Window length of waveform estimate.
    thresh: Threshold for the normalized fit.
    min_spikes: Minimum number of spikes over which we declare that a waveform
                was detected.
    interp: If True, snippets are interpolated before averaging.
    data_path: Path to data.
    save_path: Path to save results.
    verbose: If True information is printed.
  Returns:
    res: Dictionary containing:
         -electrodes: Electrodes for which a waveform was detected.
         -waveforms: Corresponding waveforms.
         -shifts: Shifts of the waveform center from the spike location.
         -fits: Fits.
  """
  for e1 in e_list_1:
    load_file = load_path[0] + str(e1) + load_path[1]
    auxload = np.load(load_file)
    chosen = auxload[1]
    print "Cells from electrode " + str(e1)
    for i in range(len(chosen)):
      if chosen[i]:
        spikes = auxload[0][i][-1][1]
        res = {"electrodes": [], "waveforms": [], "shifts": []}
        av_snip_list = []
        for e in e_list_2:
          data_file = data_path + str(e) + "_prep.npy"
          data = np.load(data_file)
          start = time.clock()
          (waveform, shift,
           av_snippets) = cell_ini(spikes, data, uf, wl_up_large, wl_up,  
                             thresh, thresh_wav, min_spikes, interp, 
                             plot_res=False, verbose=verbose)
          end = time.clock()
          print ("Electrode " + str(e) + " took " + str(end-start)) 
          av_snip_list.append([e, av_snippets])
          if waveform is not None:
            print "Waveform found"
            res["electrodes"].append(e)
            res["waveforms"].append(waveform)
            res["shifts"].append(shift)
      save_file = save_path + "_e" + str(e1) + "_" + str(i)
      np.save(save_file, [res, av_snip_list])


def cell_ini_plot_sel_script():
  electrodes = range(100,101)
  other_electrodes = range(80, 140)
  uf = 5
  wl_cell_ini = 401
  thresh_spikes = 0.35
  thresh_wav = 1
  smoothing_wl = 5
  tol = 5e-3
  load_aux = [LOAD_PATH + "uf" + str(uf),  
                "_threshspikes_" + param2str(thresh_spikes)  
                + "_threshwav_" + param2str(thresh_wav) 
                + "_tol_" + param2str(tol) + ".npy"]
  plot_path = PLOT_PATH + "/av/"
  thresh_wav_cell = 0.75
  cell_ini_plot_sel(electrodes, other_electrodes, uf, wl_cell_ini, 
                   smoothing_wl, thresh_wav_cell, load_aux, plot_path)
                   
                   
def cell_ini_plot_av_script():
  electrodes = range(100,101)
  other_electrodes = range(90,105)
  uf = 5
  wl_cell_ini = 401
  thresh_spikes = 0.35
  thresh_wav = 1
  tol = 5e-3
  load_aux = [LOAD_PATH + "uf" + str(uf),  
                "_threshspikes_" + param2str(thresh_spikes)  
                + "_threshwav_" + param2str(thresh_wav) 
                + "_tol_" + param2str(tol) + ".npy"]
  plot_path = PLOT_PATH + "/av/"
  cell_ini_plot_av(electrodes, other_electrodes, uf, wl_cell_ini, 
                   smoothing_wl, thresh_wav, load_aux, plot_path)
                     

def cell_ini_array_script():
  uf = 5
  wl_down = 41
  wl_up = uf * (wl_down - 1) + 1
  wl_up_large = 801
  tol = 5e-3
  thresh_spikes = 0.35
  thresh_wav = 1
  min_spikes = 10
  min_length = 40
  e_list_1 = [58]  # range(512)
  e_list_2 = range(40, 50)
  load_path = [LOAD_PATH + "uf" + str(uf) + "/electrode",  
                "_threshspikes_" + param2str(thresh_spikes)  
                + "_threshwav_" + param2str(thresh_wav) 
                + "_tol_" + param2str(tol) + ".npy"]
  save_path = SAVE_PATH + "res_just_av"
  interp = False
  cell_ini_array(e_list_1, e_list_2, uf, wl_up_large, wl_up, thresh_spikes, 
                 thresh_wav, min_spikes, min_length, interp, DATA_PATH, 
                 load_path, save_path, verbose=True)  
  save_path = SAVE_PATH + "res"
  interp = True
  cell_ini_array(e_list_1, e_list_2, uf, wl_up_large, wl_up, thresh_spikes, 
                 thresh_wav, min_spikes, min_length, interp, DATA_PATH, 
                 load_path, save_path, verbose=True)  


def main():
  # cell_ini_plot_av_script()
  cell_ini_plot_sel_script()


if __name__ == "__main__":
  main()