"""Evaluates data fit for a waveform and a list of locations.

Computes the normalized l2 error
"""
import math
from IPython.terminal.embed import InteractiveShellEmbed
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy
# from measurement_op import meas_op_elect
N_FIT = 5


def snippet_indices(ind_fine, uf, wl_down):
  """Indices surrounding a certain fine-grid index.

  For ind_fine = i * uf + s and wl_down = 2 * hw + 1
  the function returns indices_fine equal to
  [ind_fine - hw * uf, ind_fine + hw * uf] and
  indices_coarse equal to [i - hw, ..., i + hw] if s = 0 and
  [i - hw + 1, ..., i + hw] otherwise.

  Args:
    ind_fine: Index on the fine grid.
    uf: Upsampling factor.
    indices_coarse: Window length.

  Returns:
    indices_coarse: Indices on the coarse grid.
  """
  h_down = (wl_down - 1) / 2
  hf_fine = h_down * uf
  indices_fine = np.arange(ind_fine - hf_fine, ind_fine + hf_fine + 1)
  ind_coarse = ind_fine / uf
  shift = np.mod(ind_fine, uf)
  if shift == 0:
    indices_coarse = np.arange(ind_coarse - h_down,
                               ind_coarse + h_down + 1)
  else:
    indices_coarse = np.arange(ind_coarse - h_down + 1,
                               ind_coarse + h_down + 1)
  return indices_fine, indices_coarse


def  snippet_indices_plot(loc, uf, wl_up):
  """Computes indices to plot fit.

  Computes indices to plot fit. Locations are of the form
  loc = a + b / uf, where a is an integer and b
  is between 1 and uf

  Args:
    locations: Location of upsampled snippet.
    uf: Upsampling factor.
    wl_up: Window length.

  Returns:
    data_ind: Indices for data vector.
    waveform_ind: Indices for waveform.
    ind_coarse: Indices for plotting on coarse grid.
    ind_fine: Indices for plotting on fine grid.
  """
  h_up = (wl_up - 1)/2
  h_down = h_up / uf
  ind_data = loc / uf
  shift = np.mod(loc, uf)
  if shift == 0:
    data_ind = ind_data + np.arange(- h_down, h_down + 1)
    waveform_ind = np.arange(0, wl_up + 1, uf)
    ind_coarse = range(-h_up, h_up + 1, uf)
  else:
    data_ind = ind_data + np.arange(- h_down + 1, h_down + 1)
    # Correct shift for waveform indices
    shift = uf - shift
    waveform_ind = shift + np.arange(0, wl_up - uf, uf)
    ind_coarse = range(- h_up + shift, h_up + 1, uf)
  ind_fine = range(-h_up, h_up + 1)  
  return data_ind.astype(int), waveform_ind.astype(int), ind_coarse, ind_fine
  
  
#def evaluate_fit(x, data, waveforms, uf, wl_down):
#  """Evaluates data fit for a waveform and a list of locations.
#
#  Computes the mean-squared error, the mean-squared energy and the
#  number of active cells at each snippet.
#
#  Args:
#    x: Spike array of dimensions w x m.
#    data: Array of data of dimensions 1 x (m - 1)/uf + 1.
#    waveforms: Waveform array of dimensions w x wl_down.
#    uf: Upsampling factor.
#    wl_down: Window length.
#
#  Returns:
#    nmse: Normalized mean-squared error of the snippets corresponding
#          to each cell.
#  """
#  fit = meas_op_elect(x, waveforms, uf)
#  nmse = []
#  for i_cell in range(x.shape[0]):
#    # Locate nonzero coefficients
#    nnz_indices = np.nonzero(x[i_cell, :])[0]
#    n = len(nnz_indices)
#    nmse_cell = np.zeros(n)
#    for i in range(0, n):
#      (snip_ind_fine,
#      snip_ind_coarse) = snippet_indices(nnz_indices[i], uf,
#                                         wl_down)
#      data_snip = data[snip_ind_coarse]
#      fit_snip = fit[0, snip_ind_coarse]
#      # Compute NMSE at snippet
#      nmse_cell[i] = la.norm(data_snip - fit_snip) / la.norm(data_snip)
#    nmse.append(nmse_cell)
#  return nmse


def evaluate_fit(data, waveform, locations, wl_down, uf, loc_format="dec",
                 compute_ms=True, verbose=False, small_step=False):
  """Evaluates data fit for a waveform and a list of locations.

  Computes the mean-squared error.

  Args:
    data: Array of data.
    waveform: Upsampled waveform.
    locations: Locations of the snippets.
    wl_down: Window length.
    uf: Upsampling factor.
    compute_ms: If True, mean-square energy of snippets is returned.
  Returns:
    mse: Mean-squared error.
    data_ms: Mean-square energy.
  """
  n = len(data)
  mse = np.zeros(len(locations))
  if compute_ms:
    data_ms = np.zeros(len(locations))
  else:
    data_ms = None
  h_down = (wl_down - 1)/2
  h_up = h_down * uf
  wl_up = h_up * 2 + 1
  for i_loc in range(0, len(locations)):
    loc = locations[i_loc]
    if loc_format == "int":
      ind_data = loc / uf
      shift = loc - uf * ind_data
    else:
      # Locations are of the form loc = a + b / uf, where a is an
      # integer and b is between 1 and uf
      aux_shift, ind_data = math.modf(loc)
      shift = np.rint(aux_shift * uf)
    if shift == 0:
      data_ind = ind_data + np.arange(- h_down, h_down + 1)
      waveform_ind = np.arange(0, wl_up + 1, uf)
    else:
      data_ind = ind_data + np.arange(- h_down + 1, h_down + 1)
      waveform_ind = (uf - shift + np.arange(0, wl_up - uf, uf))
    data_samp = data[data_ind.astype("int")]
    fit = waveform[waveform_ind.astype("int")]
    # MSE
    mse[i_loc] = ((data_samp - fit) ** 2).sum()
    if verbose:
      print ("loc: " + str(loc) + " ind_data: " + str(ind_data) + " shift: " 
             + str(shift) + " fit: " + str(mse[i_loc]))
    if compute_ms: 
      data_ms[i_loc] = (data_samp ** 2).sum()
      if verbose:
        print "normsq: " + str(data_ms[i_loc])
  return mse, data_ms


def evaluate_fit_best_worst(data, waveform, locations, wl_down,
                            uf, plot_file=False):
  """Evaluates data fit for a waveform and a list of locations.

  Computes the mean-squared error.

  Args:
    data: Array of data.
    waveform: Upsampled waveform.
    locations: Locations of the snippets.
    wl_down: Window length.
    uf: Upsampling factor.

  Returns:
    Mean-squared error. Also plots fit.
  """
  mse = 0
  data_ms = 0
  best_fits = [None,None,None,None,None]
  worst_fits = [None,None,None,None,None]
  h_down = (wl_down - 1)/2
  h_up = h_down * uf
  wl_up = h_up * 2 + 1
  if plot_file:
    best_fits_MSE = np.ones((1, N_FIT))[0] * np.inf
    worst_fits_MSE = np.zeros((1, N_FIT))[0]
  for i_loc in range(0, len(locations)):
    loc = locations[i_loc]
    # Locations are of the form loc = a + b / uf, where a is an
    # integer and b is between 1 and uf
    aux_shift, ind_data = math.modf(loc)
    shift = np.rint(aux_shift * uf)
    if shift == 0:
      data_ind = ind_data + np.arange(- h_down, h_down + 1)
      waveform_ind = np.arange(0, wl_up + 1, uf)
    else:
      data_ind = ind_data + np.arange(- h_down + 1, h_down + 1)
      waveform_ind = (uf - shift +
                      np.arange(0, wl_up-uf,
                                uf))
    data_samp = data[data_ind.astype("int")]
    fit = waveform[waveform_ind.astype("int")]
    # MSE
    mse_snippet = ((data_samp - fit) ** 2).sum()
    mse += mse_snippet
    data_ms += (data_samp ** 2).sum()
    if plot_file:
      arg_max = np.argmax(best_fits_MSE)
      arg_min = np.argmin(worst_fits_MSE)
      if best_fits_MSE[arg_max] > mse_snippet:
        best_fits_MSE[arg_max] = mse_snippet
        best_fits[arg_max] = [data_samp, fit, i_loc]
      if worst_fits_MSE[arg_min] < mse_snippet:
        worst_fits_MSE[arg_min] = mse_snippet
        worst_fits[arg_min] = [data_samp, fit, i_loc]
  n_snippets = len(locations)
  mse = mse / n_snippets
  data_ms = data_ms / n_snippets
  return mse, data_ms, best_fits, worst_fits


def snippet_indices_test():
  """Test for snippet_indices."""
  fine_val = [3, 101, 200, 1001, 1200]
  uf_val = [2, 5, 5, 10, 10]
  wl_val = [3, 21, 41, 31, 41]
  coarse_exp = [[1, 2], range(11, 31), range(20, 61), range(86, 116),
                range(100, 141)]
  fine_exp = [range(1, 6), range(51, 152), range(100, 301), range(851, 1152),
              range(1000, 1401)]
  for i in range(0, len(fine_val)):
    (res_fine,
     res_coarse) = snippet_indices(fine_val[i], uf_val[i], wl_val[i])
    print "Test fine indices"
    print "Expected: " + str(fine_exp[i])
    print " Result: " + str(res_fine)
    if (len(res_fine) != len(fine_exp[i]) or
        (res_fine - fine_exp[i]).sum() != 0):
      print "WRONG"
    else:
      print "ALRIGHT!"
    print "Test coarse indices"
    print "Expected: " + str(coarse_exp[i])
    print " Result: " + str(res_coarse)
    if (len(res_coarse) != len(coarse_exp[i]) or
        (res_coarse - coarse_exp[i]).sum() != 0):
      print "WRONG"
    else:
      print "ALRIGHT!"
