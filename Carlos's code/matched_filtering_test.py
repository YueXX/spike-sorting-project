import os.path
import time
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
import numpy as np
import local_directories as ldir
from numpy import linalg as la
from preprocess_data import preprocess_data
from matched_filtering import *
from data_model import *


def matched_filtering_electrode(data, waveforms, uf, thresh, verbose=False):
  """Fit isolated spikes for a whole electrode.

  The fit is obtained by computing the fit between the data and
  undersampled versions of the waveform.

  Args:
    data: Data.
    waveforms: Waveforms corresponding to each cell.
    uf: Upsampling factor.
    thresh: Threshold on the normalized root mean-squared error.
  Returns:
    loc: Spike locations.
    fit: Normalized fit at spike locations.
  """
  loc = []
  fit = []
  wl_down = (waveforms.shape[1] - 1) / uf + 1
  if verbose:
    start = time.clock()
  normsq = normsq_data(data, uf, wl_down)
  if verbose:
    end = time.clock()
    print "normsq takes: " + str(end-start)
  for i_cell in range(waveforms.shape[0]):
    res, spike_fit = matched_filtering(data, waveforms[i_cell, :], normsq, 
                                       uf, thresh, indices, verbose)
    loc.append(res)
    fit.append(spike_fit)
  return loc, fit
  
  
def normsq_data_test():
  """Tests normsq_data.
  """
  print "Testing normsq_data"
  data = np.sqrt(np.array([1.0, 2.0, 1.5, 4.5, 3.0]))
  wl_down = 3
  uf = 3
  res_expected = np.array([1.0, 1.0, 1.0, 4.5, 3.5, 3.5, 8.0, 6.0, 6.0, 9.0,
                           1.0, 1.0, 1.0])
  res = normsq_data(data, uf, wl_down)
  npt.assert_allclose(res_expected, res, err_msg="Test 1 failed")
  
  
def normsq_wav_test():
  """Tests normsq_wav.
  """
  print "Testing locate_snippets_segment on simulated data"
  waveform = np.array([1.0, 2.0, 1.5, 4.5, 3.0, 3.5, 8.0, 6.0, 4.5, 9.0])
  uf = 3
  n = 10
  res_expected = np.zeros(uf * (n-1) + 1)
  for i in range(n-1):
    res_expected[i * uf] = 1.0 + 4.5**2 + 8.0**2 + 9.0**2
    res_expected[i * uf + 1] = 2.0**2 + 3.0**2 + 6.0**2
    res_expected[i * uf + 2] = 1.5**2 + 3.5**2 + 4.5**2
  res_expected[-1] = res_expected[0]
  res = normsq_wav(waveform, uf, n)
  npt.assert_allclose(res_expected, res, err_msg="Test 1 failed")
  print "Tests completed"
  
  
def inner_prod_upsampled_test(uf, wl_down, n):
  """ Test for matched filtering.
  Args:
    uf : Upsampling factor.
    wl_down: Waveform length.
    n: Length of data."""
  print ("inner_prod_upsampled(upsampling factor = " + str(uf)
         + ", wl_down = " + str(wl_down) + ", n = "
         + str(n) + ")")
  wl_up = (wl_down - 1) * uf + 1
  h = (wl_down - 1) / 2
  h_up = (wl_up - 1) / 2
  waveform = np.random.randn(wl_up)
  data = np.random.randn(n)
  norm_fit = 2 * np.ones((n - 1) * uf + 1)
  for i in range(h, n - h):
    for i_up in range(uf):
      # Upsampled index
      ind = i * uf + i_up
      if i_up == 0:
        y_i = data[(i - h):(i + h + 1)]
        wav_i = waveform[::uf]
      else:
        y_i = data[(i - h + 1):(i + h + 1)]
        # Note that the shift is uf - i_up
        wav_i = waveform[(uf - i_up)::uf]
      norm_fit[ind] = la.norm(y_i - wav_i) / la.norm(y_i)
  norm_fit[:h_up] = 1
  norm_fit[-h_up:] = 1
  normsq = normsq_data(data, uf, wl_down)
  inner_prod = inner_prod_upsampled(data, waveform, uf)
  n = len(data)
  wav_normsq = normsq_wav(waveform, uf, n)
  aux_norm_fit = 1 + (wav_normsq - 2 * inner_prod) / normsq
  norm_fit_res = np.sqrt(aux_norm_fit)
  norm_fit_res[:h_up] = 1
  norm_fit_res[-h_up:] = 1
  norm_diff = la.norm(norm_fit - norm_fit_res)
  if norm_diff < 1e-8:
    print "Success! Difference: " + str(norm_diff)
  else:
    print "Failed! Difference: " + str(norm_diff)
    print norm_fit - norm_fit_res
    print "Expected:"
    print norm_fit
    print "Obtained"
    print norm_fit_res


def matched_filtering_sim():
  """Test for matched_filtering with simulated data.
  """
  print ("matched_filtering_sim()")
  uf = 5
  m = 50001
  n_cells = 3
  thresh = 0.1
  spiking_rate = 1. / 5000
  refractory_period = 40 * uf
  # Simulate spike train
  x = np.zeros((n_cells, m))
  for i_cell in range(0, n_cells):
    x[i_cell, :] = simulate_spike_train(m, spiking_rate,
                                        refractory_period)
  # Load waveforms
  wl_down = 41
  e = 58
  wav_file = ("/Users/cfgranda/Google Drive" +
              "/spike_sorting_results/clustering/" +
              "uf" + str(uf) + "wl" + str(wl_down) + "/no_proj/av/k3/electrode"
              + str(e) + "_minmax.npz")
  auxload = np.load(wav_file)
  waveforms = auxload["wav"]
  # ipshell = InteractiveShellEmbed()
  # ipshell()
  y = conv_wav_spikes(x, waveforms, uf)
  data = y[0] + 0.0001 * np.random.randn(len(y[0]))
  spike_locations, fit = matched_filtering_electrode(data, waveforms, uf, thresh)
  ind_coarse = range(0, (len(data) - 1) * uf + 1, uf)
  plt.ion()
  plt.figure(figsize=(35.0, 15.0))
  h = (waveforms.shape[1] - 1)/2
  colors = "kbrg"
  plt.plot(ind_coarse, data, "--bo")
  for i_cell in range(waveforms.shape[0]):
    for spike in spike_locations[i_cell]:
      ind_fine = range(spike - h, spike + h + 1)
      plt.plot(ind_fine, waveforms[i_cell, :], "--x" + colors[i_cell])
  
  
def matched_filtering_test_real(e, thresh, n):
  """Test for fit_isolated_spikes with real data.
  """
  uf = 5
  data_file = DATA_FILE + str(e) + ".npy"
  data_electrode = np.load(data_file)
  data = preprocess_data(data_electrode)
  data = data[:n]
  wl_down = 41
  # Load waveforms
  wav_file = ("/Users/cfgranda/Google Drive" +
              "/spike_sorting_results/clustering/" +
              "uf" + str(uf) + "wl" + str(wl_down) + "/no_proj/av/k3/electrode"
              + str(e) + "_minmax.npz")
  auxload = np.load(wav_file)
  waveforms = auxload["wav"]
  loc_all, fit_all = matched_filtering_electrode(data, waveforms, uf, thresh)
  ind_coarse = range(0, (len(data) - 1) * uf + 1, uf)
  plt.ion()
  plt.figure(figsize=(35.0, 15.0))
  h = (waveforms.shape[1] - 1)/2
  colors = "krgmcb"
  plt.plot(ind_coarse, data, "--bo")
  for i_cell in range(waveforms.shape[0]):
    for spike in loc_all[i_cell]:
      ind_fine = range(spike - h, spike + h + 1)
      plt.plot(ind_fine, waveforms[i_cell, :], "--x" + colors[i_cell])
  plt.figure(figsize=(35.0, 15.0))
  for i_cell in range(waveforms.shape[0]):
    plt.plot(np.sort(fit_all[i_cell]), "--o" + colors[i_cell])
  ind_ini = range(0, int(n), int(n / 10))
  ind_end = ind_ini[1:] + [n]
  indices = [ind_ini[::2], ind_end[::2]]
  print (np.array(indices) * uf)
  loc, fit = matched_filtering_electrode(data, waveforms, uf, thresh, indices)
  plt.ion()
  plt.figure(figsize=(35.0, 15.0))
  h = (waveforms.shape[1] - 1)/2
  colors = "krgmcb"
  plt.plot(ind_coarse, data, "--bo")
  print "All spikes"
  print loc_all
  print "Spikes in chunks"
  print loc
  for i_cell in range(waveforms.shape[0]):
    for spike in loc[i_cell]:
      ind_fine = range(spike - h, spike + h + 1)
      plt.plot(ind_fine, waveforms[i_cell, :], "--x" + colors[i_cell])
  plt.figure(figsize=(35.0, 15.0))
  for i_cell in range(waveforms.shape[0]):
    plt.plot(np.sort(fit[i_cell]), "--o" + colors[i_cell])
  

def matched_filtering_test_speed(e, thresh, n_val):
  """Test for fit_isolated_spikes with real data.
  """
  uf = 5
  data_file = DATA_FILE + str(e) + ".npy"
  data_electrode = np.load(data_file)
  data = preprocess_data(data_electrode)
  wl_down = 41
  # Load waveforms
  wav_file = ("/Users/cfgranda/Google Drive" +
              "/spike_sorting_results/clustering/" +
              "uf" + str(uf) + "wl" + str(wl_down) + "/no_proj/av/k3/electrode"
              + str(e) + "_minmax.npz")
  auxload = np.load(wav_file)
  waveforms = auxload["wav"]
  for n in n_val:
    data_e = data[:n]
    print "Length of data vector: " + str(len(data_e))
    start = time.clock()
    matched_filtering_electrode(data_e, waveforms, uf, thresh, verbose=True)
    end = time.clock()
    print "Matched filter takes : " + str(end-start) + " for n = " + str(n)

    
def main():
  plt.close("all")
  test_normsq = True
  test_ip_upsampled = False
  test_mf_sim = False
  test_mf_real = False
  test_speed = False
  if test_normsq: 
    normsq_data_test()
    normsq_wav_test()
  wl_down = 5
  uf = 3
  thresh = 0.4
  electrodes = [56, 58, 61] 
  if test_ip_upsampled:
    n = 20
    inner_prod_upsampled_test(uf, wl_down, n)
  if test_mf_sim:  
    matched_filtering_sim()
  if test_mf_real:
    for e in electrodes:
      n = 1e4
      matched_filtering_test_real(e, thresh, n)   
  if test_speed:  
    n_val = [1e4, 1e5, 1.2e6]
    for e in electrodes:
      matched_filtering_test_speed(e, thresh, n_val)
  
  
if __name__ == "__main__":
  main()   