"""Segments data according to the activity (high energy blocks) over
   a certain period. If the energy of a given window is above a threshold
   the block is extracted and set aside. The threshold is set such that
   the total number of samples are in [min_fraction,max_fraction] x total.
   The functions here are called by upsample_snippets function."""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
#from IPython.terminal.embed import InteractiveShellEmbed


### AUXILIARY FUNCTIONS ###


def apply_thresh(x, wl, thresh):
  """Thresholds x.

  Selects points which are within a window that is completely above threshold.

  Args:
    x: Array to be thresholded.
    window_length: Window length, must be odd.
    threshold: Threshold.

  Returns:
    ini_ind: Initial indices of selected segments.
    end_ind: Final indices of selected segments.
  """
  aux_thresh = x > thresh
  h = (wl - 1) / 2
  h = int(h)
  selected = aux_thresh.copy()
  for i in np.concatenate((range(-h, 0), range(1, h + 1))):
    selected = (np.logical_and(selected, np.roll(aux_thresh, i)))
  # An entry in selected is True if it is above threshold and it is
  # surrounded by h entries that are above threshold on both sides
  ini_ind = []
  end_ind = []
  above_thresh = False
  # We iterate over all the entries
  for i in range(len(selected)):
  # We check if the previous entry was above threshold or not
    if above_thresh:
      # Transition to below threshold, period of activity ends
      if not selected[i]:
        end_ind.append(i)
        above_thresh = False
    else:
      # Transition to above threshold, period of activity begins
      if selected[i]:
        ini_ind.append(i)
        above_thresh = True
  if above_thresh:
    end_ind.append(len(selected) - 1)
  return np.array(ini_ind), np.array(end_ind)


### FUNCTIONS ###


def segment_data(data, wl, max_fraction, min_fraction, plot_res=False):
  """Segments data by computing local energy and thresholding.

  Computes norm over windows of length wl and then thresholds.

  Args:
    data: Data.
    wl: Window length, must be odd.
    max_fraction: Upper bound on the energy of the snippets.
    min_fraction: Lower bound on the energy of the snippets.
    plot_res: If True, results are plotted.

  Returns:
    ini_ind: Initial indices of selected segments.
    end_ind: Final indices of selected segments.
  """
  n = len(data)
  h = (wl - 1) / 2
  h = int(h)
  windowed_data_sq = np.convolve(data ** 2, np.ones(h + 1), "same")
  windowed_data_aux = np.sqrt(windowed_data_sq)
  windowed_data = windowed_data_aux / np.amax(windowed_data_aux)
  if plot_res:
    plot_n = 100000
    plt.figure(figsize=(35.0, 15.0))
    plt.plot(data[:plot_n], label="Data")
    plt.plot(windowed_data[:plot_n], label="Windowed data")
    plt.legend()
  thresh_lower = 0.0
  thresh_upper = 1.0
  thresh = 0.5
  while True:
    ini_ind, end_ind = apply_thresh(windowed_data, wl, thresh)
    fraction = float(sum(end_ind - ini_ind)) / n
    if fraction < min_fraction:
      thresh_upper = thresh
    elif fraction > max_fraction:
      thresh_lower = thresh
    else:
      break
    thresh = (thresh_upper + thresh_lower) / 2.0
    #print("Threshold: " + str(thresh) + " Fraction: " + str(fraction))
  #print("Fraction: " + str(fraction))
  if plot_res:
    aux_legend = True
    for i in range(len(ini_ind)):
      if ini_ind[i] > plot_n:
        break
      if aux_legend:
        plt.plot(range(ini_ind[i],end_ind[i]), data[ini_ind[i]:end_ind[i]],
                 "rx", label="Selected")
        aux_legend = False
      else: 
        plt.plot(range(ini_ind[i],end_ind[i]), data[ini_ind[i]:end_ind[i]],
                 "rx")
    plt.legend()
  return ini_ind, end_ind