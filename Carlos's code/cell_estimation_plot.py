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
from para_fun import param2str

DATA_PATH = "/Users/starry1990/Documents/spike_sorting_real_data/data/electrode"
PLOT_PATH = "/Users/starry1990/Documents/spike_sorting_real_data/plots/"
UPSAMPLE_DIR = "/Users/starry1990/Documents/spike_sorting_real_data/upsampled_snippets/"
LOAD_PATH = " /Users/starry1990/Documents/spike_sorting_real_data/clustering/"

# DATA_PATH = "/Users/cfgranda/Google Drive/spike_sorting/data/electrode"
# WAV_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/clustering/"
# PLOT_PATH =  "/Users/cfgranda/Google Drive/spike_sorting_results/plots/cell_ini"
# LOAD_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/wav_est/"
# SAVE_PATH = "/Users/cfgranda/Google Drive/spike_sorting_results/cell_est/ini/"
# DATA_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting/data/electrode"
# WAV_PATH = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/clustering/"
# SAVE_PATH =  "/usr/local/google/home/cfgranda/Google Drive/spike_sorting_results/wav_est/"


def cell_ini_plot_sel(electrodes, other_electrodes, uf, wl_cell_ini, 
                      smoothing_wl, thresh_wav, load_aux, plot_path):
  """Plots average of data around the spikes at other electrodes.
  Averages snippets around the spikes and plots them.
  Args:
    electrodes: List of electrodes.
    other_electrodes: Electrodes at which we perform the averaging. 
    uf: Upsampling factor.
    wl_cell_ini: Window over which we average.
  """
  for e1 in electrodes:
    plot_dir = PLOT_PATH + "/e" + str(e1)
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    load_file = load_aux[0] + "/electrode" + str(e1) + load_aux[1]
    auxload = np.load(load_file)
    chosen = auxload[1]
    print "Cells from electrode " + str(e1)
    for e in other_electrodes:
      print "Electrode " + str(e)
      data_file = DATA_PATH + str(e) + "_prep.npy"
      data = np.load(data_file)
      aux_plot = []
      for i in range(len(chosen)):
        if chosen[i]:
          spikes = auxload[0][i][-1][1] 
          start = time.clock()
          av_interp = average_waveforms_interp_fast(spikes, data, uf, 
                                                    wl_cell_ini)
          end = time.clock() 
          print ("Averaging + interpolation takes " + str(end-start)) 
          start = time.clock()
          err_rel_av_interp_l1 = std_waveforms_norm(av_interp, spikes, data, 
                                                    uf, "l1")
          err_rel_av_interp_l1[err_rel_av_interp_l1 > 3] = 3
          end = time.clock()
          print ("Computing l1-std takes " + str(end-start)) 
          start = time.clock()
          err_interp_smoothed = std_waveforms_norm(av_interp, spikes, data, 
                                                   uf, "l1", 
                                                   smoothing_wl = smoothing_wl)
          err_interp_smoothed[err_interp_smoothed > 3] = 3
          end = time.clock()
          print ("Computing smoothed l1-std takes " + str(end-start)) 
          w_trim, shift = trim_waveform(av_interp, err_rel_av_interp_l1, uf, 
                                   thresh_wav)
          w_trim_smoothed, shift_smoothed = trim_waveform(av_interp, 
                                                          err_interp_smoothed, 
                                                          uf, thresh_wav)
          aux_plot.append([av_interp, err_rel_av_interp_l1, w_trim, shift, 
                           err_interp_smoothed, w_trim_smoothed, 
                           shift_smoothed])
      if len(aux_plot) > 0:
        fig = plt.figure(figsize=(35.0, 15.0))
        # plt.title("Electrode " + str(e))
        rows = len(aux_plot[0]) - 2
        cols = len(aux_plot)
        for w in range(len(aux_plot)):
          ax = fig.add_subplot(rows, cols, w + 1)
          plt.plot(aux_plot[w][0], ".")
          ax.set_title("Waveform " + str(w + 1))
          ax = fig.add_subplot(rows, cols, cols + w + 1)
          plt.plot(aux_plot[w][1], ".")
          ax = fig.add_subplot(rows, cols,  2 * cols + w + 1)
          plt.plot(aux_plot[w][2], ".")
          ax.text(0, 0, "Shift: " + str(aux_plot[w][3]),
                  horizontalalignment="left", verticalalignment="bottom",
                  transform=ax.transAxes)
          ax = fig.add_subplot(rows, cols, 3 * cols + w + 1)
          plt.plot(aux_plot[w][4], ".")
          ax = fig.add_subplot(rows, cols, 4 * cols + w + 1)
          plt.plot(aux_plot[w][5], ".")
          ax.text(0, 0, "Shift: " + str(aux_plot[w][6]),
                horizontalalignment="left", verticalalignment="bottom",
                transform=ax.transAxes)
        plot_file = plot_dir + "/e" + str(e) + ".jpg"
        fig.savefig(plot_file)
        plt.close(fig)


def cell_ini_plot_av(electrodes, other_electrodes, uf, wl_cell_ini, 
                     smoothing_wl, thresh_wav, load_aux, plot_path):
  """Plots average of data around the spikes at other electrodes.
  Averages snippets around the spikes and plots them.
  Args:
    electrodes: List of electrodes.
    other_electrodes: Electrodes at which we perform the averaging. 
    uf: Upsampling factor.
    wl_cell_ini: Window over which we average.
  """
  for e1 in electrodes:
    plot_dir = PLOT_PATH + "/e" + str(e1)
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    load_file = load_aux[0] + "/electrode" + str(e1) + load_aux[1]
    auxload = np.load(load_file)
    chosen = auxload[1]
    print "Cells from electrode " + str(e1)
    for i in range(len(chosen)):
      if chosen[i]:
        spikes = auxload[0][i][-1][1]
        for e in other_electrodes:
          print "Electrode " + str(e)
          data_file = DATA_PATH + str(e) + "_prep.npy"
          data = np.load(data_file)
          start = time.clock()
          av, n_shift = average_waveforms(spikes, data, uf, wl_cell_ini)
          end = time.clock()
          print ("Averaging takes " + str(end-start)) 
          start = time.clock()
          av_interp = average_waveforms_interp_fast(spikes, data, uf, 
                                                    wl_cell_ini)
          end = time.clock() 
          print ("Averaging + interpolation takes " + str(end-start)) 
          start = time.clock()
          err_rel_av_l2 = std_waveforms_norm(av, spikes, data, uf, "l2")
          err_rel_av_l2[err_rel_av_l2 > 3] = 3
          end = time.clock()
          print ("Computing l2-std takes " + str(end-start))
          start = time.clock()
          err_rel_av_l1 = std_waveforms_norm(av, spikes, data, uf, "l1")
          err_rel_av_l1[err_rel_av_l1 > 3] = 3
          end = time.clock()
          print ("Computing l1-std takes " + str(end-start))
          start = time.clock()
          err_rel_av_interp_l2 = std_waveforms_norm(av_interp, spikes, data, 
                                                    uf, "l2")
          err_rel_av_interp_l2[err_rel_av_interp_l2 > 3] = 3
          end = time.clock()
          print ("Computing l2-std takes " + str(end-start)) 
          start = time.clock()
          err_rel_av_interp_l1 = std_waveforms_norm(av_interp, spikes, data, 
                                                    uf, "l1")
          err_rel_av_interp_l1[err_rel_av_interp_l1 > 3] = 3
          end = time.clock()
          print ("Computing l1-std takes " + str(end-start)) 
          start = time.clock()
          err_smoothed = std_waveforms_norm(av_interp, spikes, data, uf, "l1", 
                                                   smoothing_wl = smoothing_wl)
          err_smoothed[err_smoothed > 3] = 3
          end = time.clock()
          print ("Computing smoothed l2-std takes " + str(end-start)) 
          start = time.clock()
          err_interp_smoothed = std_waveforms_norm(av_interp, spikes, data, 
                                                   uf, "l2", 
                                                   smoothing_wl = smoothing_wl)
          err_interp_smoothed[err_interp_smoothed > 3] = 3
          end = time.clock()
          print ("Computing smoothed l1-std takes " + str(end-start)) 
          fig = plt.figure(figsize=(35.0, 15.0))
          ax = fig.add_subplot(2, 4, 1)
          plt.plot(av, ".")
          ax.set_ylabel("Averaging")
          ax = fig.add_subplot(2, 4, 2)
          plt.plot(err_rel_av_l2, ".")
          ax.set_title("Normalized l2 -std")
          ax = fig.add_subplot(2, 4, 3)
          plt.plot(err_rel_av_l1, ".")
          ax.set_title("Normalized l1 -std")
          ax = fig.add_subplot(2, 4, 4)
          plt.plot(err_smoothed, ".")
          ax.set_title("Smoothed l1 -std")
          ax = fig.add_subplot(2, 4, 5)
          plt.plot(av_interp, ".")
          ax.set_ylabel("Averaging + interpolation")
          ax = fig.add_subplot(2, 4, 6)
          plt.plot(err_rel_av_interp_l2, ".")
          ax = fig.add_subplot(2, 4, 7)
          plt.plot(err_rel_av_interp_l1, ".")
          ax = fig.add_subplot(2, 4, 8)
          plt.plot(err_interp_smoothed, ".")
          plot_file = plot_dir + "/w" + str(i) + "e" + str(e) + ".jpg"
          fig.savefig(plot_file)
          plt.close(fig)
      

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