"""Test for preprocess_data.py.
"""
# from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from preprocess_data import *
import local_directories as ldir

def filter_data_plot(params=None):
  """Plot filtering results.

  Args:
    params: Filtering paramenters.
  """
  if params is None:
    params = default_params()
  nyq = 0.5 * params["f_samp"]
  f_low = params["f_lp"] / nyq
  if params["filter_type"] == "butter":
    b, a = signal.butter(params["order"], f_low, btype="high")
  plt.ion()
  w, h = signal.freqz(b, a, worN=2000)
  plt.loglog((nyq / np.pi) * w, abs(h), label=params["filter_type"])
  

def filter_data_test(electrode):
  """Tests filtering on electrode data.

  Args:
    electrode: Electrode to be tested.
  """
  data = np.load(ldir.DATA_PATH + str(electrode) + ".npy")
  f_samp = 2e4
  f_low = 100
  ind_ini = 10000
  ind_end = 20000
  params_1 = {"filter_type": "butter", "order": 2, "f_lp": f_low,
              "f_samp": f_samp, "padlen": 1e4}
  params_2 = {"filter_type": "butter", "order": 4, "f_lp": f_low,
              "f_samp": f_samp, "padlen": 1e4}
  params_3 = {"filter_type": "butter", "order": 6, "f_lp": f_low,
              "f_samp": f_samp, "padlen": 1e4}
  params_4 = {"filter_type": "butter", "order": 8, "f_lp": f_low,
              "f_samp": f_samp, "padlen": 1e4}
  filtered_1 = filter_data(data, params_1)
  filtered_2 = filter_data(data, params_2)
  filtered_3 = filter_data(data, params_3)
  filtered_4 = filter_data(data, params_4)
  # Plot residual
  plt.ion()
  plt.figure(figsize=(35.0, 15.0))
  plt.plot(data[ind_ini:ind_end] - filtered_1[ind_ini:ind_end],
           "--xb", label="Butter 2")
  plt.plot(data[ind_ini:ind_end] - filtered_2[ind_ini:ind_end],
           "--dk", label="Butter 4")
  plt.plot(data[ind_ini:ind_end] - filtered_3[ind_ini:ind_end],
           "--.g", label="Butter 6")
  plt.plot(data[ind_ini:ind_end] - filtered_4[ind_ini:ind_end],
           "--oc", label="Butter 8")
  plt.legend()
  plt.title("Residuals (electrode " + str(electrode) + ")")
  # Plot filtered data
  plt.figure(figsize=(35.0, 15.0))
  plt.plot(data[ind_ini:ind_end], "--xb", label="Data")
  plt.plot(filtered_1[ind_ini:ind_end], "--dr", label="Butter 2")
  plt.plot(filtered_2[ind_ini:ind_end], "--dk", label="Butter 4")
  plt.plot(filtered_3[ind_ini:ind_end], "--dg", label="Butter 6")
  plt.plot(filtered_4[ind_ini:ind_end], "--dc", label="Butter 8")
  plt.legend()
  plt.title("Electrode " + str(electrode))
  # Plot spectrum
  plt.figure(figsize=(35.0, 15.0))
  plt.plot(np.absolute(np.fft.fft(data)), "xb", label="Data")
  plt.plot(np.absolute(np.fft.fft(filtered_1)), "dr", label="Butter 2")
  plt.plot(np.absolute(np.fft.fft(filtered_2)), "dk", label="Butter 4")
  plt.plot(np.absolute(np.fft.fft(filtered_2)), "dg", label="Butter 6")
  plt.plot(np.absolute(np.fft.fft(filtered_2)), "dc", label="Butter 8")
  plt.legend()
  plt.title("Spectrum (electrode " + str(electrode) + ")")
  

def preprocess_data_test(electrode, params=None):
  """Plot intermediate steps in preprocessing data.

  Args:
    electrode: Electrode.
    params: Filtering parameters.
  """
  data = np.load(ldir.DATA_PATH + str(electrode) + ".npy")
  ind_ini = 10000
  ind_end = 20000
  filtered = filter_data(data, params)
  res = filtered / np.amax(np.absolute(filtered))
  plt.ion()
  plt.figure(figsize=(35.0, 15.0))
  plt.plot(data[ind_ini:ind_end], "--xb", label="Data")
  plt.plot(filtered[ind_ini:ind_end], "--dr", label="Filtered")
  plt.plot(res[ind_ini:ind_end], "--xg", label="Normalized")
  plt.legend()
  plt.title("Electrode " + str(electrode))

def main():
  plt.close("all")
  electrodes = [120]
  for e in electrodes:
    filter_data_test(e)
    preprocess_data_test(e)
  

if __name__ == "__main__":
  main()
