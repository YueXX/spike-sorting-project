"""Preprocesses electrode data.

Perform high-pass filtering and normalizing.

See:

http://www.buzsakilab.com/content/PDFs/HarrisJNeurophys2000.pdf
http://www.scholarpedia.org/article/Spike_sorting
"""
import numpy as np
import scipy.signal as signal


### AUXILIARY FUNCTIONS ###


def default_params():
  """Set default parameters for filtering.

  Returns:
    params: Default parameters.
  """
  f_samp = 2e4
  f_low = 100
  params = {"filter_type": "butter", "order": 6, "f_lp": f_low,
              "f_samp": f_samp, "padlen": 1e4}
  return params


### FUNCTIONS ###


def filter_data(data, params=None):
  """Filters electrode data.

  Args:
    data: Array of data.
    params: Filtering paramenters.
  Returns:
    res: Filtered data.
  """
  if params is None:
    params = default_params()
  nyq = 0.5 * params["f_samp"]
  f_low = params["f_lp"] / nyq
  # f_high = params["f_hp"] / nyq
  if params["filter_type"] == "butter":
    b, a = signal.butter(params["order"], f_low, btype="high")
    # [f_low, f_high], #  btype="band")
    res = signal.filtfilt(b, a, data, padlen=params["padlen"])
  return res


def preprocess_data(data, params=None):
  """Preprocess data by high-pass filtering, centering and normalizing.

  Args:
    data: Array of data.
    params: Filtering paramenters.
  Returns:
    res: Preprocessed data.
  """
  filtered = filter_data(data, params)
  centered = filtered - np.mean(filtered)
  res = centered / np.amax(np.absolute(centered))
  return res
  
