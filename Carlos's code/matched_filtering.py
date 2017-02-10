"""Locates regions that are well approximated by a single spike.

A local fit is computed and used to determine the locations of single
spiking events.
"""
import time
import numpy as np
from numpy import linalg as la
from IPython.terminal.embed import InteractiveShellEmbed


def inner_prod_upsampled(data, waveform, uf):
  """Computes inner product between the data and a waveform.

  Args:
    data: Data (on coarse grid).
    waveform: Waveform on fine grid.
    uf: Upsampling factor.
  Returns:
    inner_prod: Inner product with a waveform for each point on the
                fine grid.
  """
  wl_up = len(waveform)
  assert np.mod(wl_up, 2) == 1, ("Waveform length should be odd but it's "
                              + str(wl_up))
  n = len(data)
  n_upsamp = (n - 1) * uf + 1
  inner_prod = np.zeros(n_upsamp)
  # We flip the waveform to implement the inner product via convolution
  inner_prod[::uf] = np.convolve(data, waveform[::-uf], "same")
  for i in range(1, uf):
    aux_wav = waveform[(uf - i)::uf]
    inner_prod[i::uf] = np.convolve(data, aux_wav[::-1], "same")[1::]
  h = (wl_up - 1)/2
  return inner_prod


def normsq_data(data, uf, wl_down):
  """Computes the squared norm per entry.

  Computes squared norm of the window surrounding the entry. For entries that
  are a multiple of uf it is the sum of wl_down squared entries. For the rest
  it is the sum of wl_down - 1 entries.

  Args:
    data: Data (on coarse grid).
    uf: Upsampling factor.
    wl_down: Window length.
  Returns:
    normsq: Squared norm.
  """
  aux_norm = np.convolve(data ** 2, np.ones(wl_down - 1), "same")
  normsq = np.append(np.repeat(aux_norm[1:], uf), np.array([1])) 
  normsq[::uf] = np.convolve(data ** 2, np.ones(wl_down), "same")
  wl_up = (wl_down - 1) * uf + 1
  h = (wl_up - 1)/2
  normsq[:h] = 1
  normsq[-h:] = 1
  return normsq
  
  
def normsq_wav(waveform, uf, n):
  """ Squared norm of the downsampled waveform for different shifts.

  The norm for each of the shifts is tiled so that it can be combined with the
  result of normsq_data.
  
  Args:
    waveform: Waveform (on fine grid).
    uf: Upsampling factor.
    wl_down: Window length.
  Returns:
    normsq: Squared norm.
  """
  normsq_aux = np.zeros(uf)
  normsq_aux[0] = la.norm(waveform[::uf])**2
  for i in range(1, uf):
    normsq_aux[i] = la.norm(waveform[(uf - i)::uf])**2
  normsq = np.append(np.tile(normsq_aux, n - 1), normsq_aux[0])
  return normsq
  
  
def matched_filtering_chunk(data, waveform, uf, thresh, normsq=None, 
                            verbose=False):
  """Fit spikes on a chunk of data.

  The fit is obtained by computing the fit between the data and
  undersampled versions of the waveform.

  Args:
    data: Data.
    waveform: Waveform.
    uf: Upsampling factor.
    thresh: Threshold on the normalized root mean-squared error.
  
  Returns:
    loc: Spike locations.
    spike_fit: Normalized fit at spike locations.
  """
  wl_up = len(waveform)
  h_up = (wl_up - 1) / 2
  wl_down = 2 * h_up / uf + 1
  if verbose:
    start = time.clock()
  inner_prod = inner_prod_upsampled(data, waveform, uf)
  if verbose:
    end = time.clock()
    print "Inner product takes: " + str(end-start)
  n = len(data)
  if normsq is None:
    normsq = normsq_data(data, uf, wl_down)
  wav_normsq = normsq_wav(waveform, uf, n)
  aux_norm_fit = 1 + (wav_normsq - 2 * inner_prod) / normsq
  norm_fit = np.sqrt(aux_norm_fit)
  if verbose:
    start = time.clock()
  # This is to avoid local minima with same values (probably not needed)
  epsilon = 1e-9
  aux_fit = norm_fit + epsilon * np.random.rand(len(norm_fit))
  # This was quite slow:
#  aux_fit[aux_fit > thresh] = 10
#  # We ignore the beginning and the end
#  local_minima = np.ones(len(aux_fit))
#  # Each point is compared to all the other points within a window of width
#  # wl_up, local_extrema will be True at that point only if the
#  # comparison is True for all the points
#  for i in np.concatenate((range(-h, 0), range(1, h + 1))):
#    local_minima = (np.logical_and(local_minima,
#                                   aux_fit < np.roll(aux_fit, i)))
#  aux_loc = np.nonzero(local_minima)[0]
  counter = 0
  min_val = 10
  min_loc = 0
  loc = []
  # min_val is the running minimum of the normalized fit 
  # Look for normalized-fits that are above threshold
  for i in range(h_up, len(normsq) - h_up):
    val = aux_fit[i]
    if val <= thresh and val < min_val:
      min_val = val
      min_loc = i
      counter = 0
    elif min_val < 10:
      counter += 1
    if counter == h_up:
        loc.append(min_loc)
        min_val = 10
        counter = 0
  loc = np.array(loc)
  if verbose:
    end = time.clock()
    print "Local minima search takes: " + str(end-start)
  if len(loc) > 0:
    spike_fit = norm_fit[loc]
  else:
    spike_fit = np.array([])
  return loc, spike_fit


def matched_filtering(data, waveform, normsq, uf, thresh, indices=None,
                      verbose=False):
  """Fit spikes.

  The fit is obtained by computing the fit between the data and
  undersampled versions of the waveform.

  Args:
    data: Data.
    waveform: Waveform.
    uf: Upsampling factor.
    thresh: Threshold on the normalized root mean-squared error.
    indices: Contains initial and final indices for the chunks of data on which
             we want to apply matched filtering.
    
  Returns:
    loc: Spike locations.
    spike_fit: Normalized fit at spike locations.
  """
  if indices is None:
    loc, spike_fit = matched_filtering_chunk(data, waveform, normsq, uf, 
                                             thresh, verbose)
  else:
    ind_ini = indices[0]
    ind_end = indices[1]
    loc = np.array([])
    spike_fit = np.array([])
    for i in range(len(ind_ini)):
      if verbose:
        print "Chunk: " + str(ind_ini[i]) + ":" + str(ind_end[i])
      chunk = data[ind_ini[i]:ind_end[i]]
      ind_ini_up = ind_ini[i] * uf
      ind_end_up = (ind_end[i] - 1) * uf + 1
      normsq_chunk = normsq[ind_ini_up:ind_end_up]
      loc_chunk, fit_chunk = matched_filtering_chunk(chunk, waveform, 
                                                     normsq_chunk, uf, thresh, 
                                                     verbose)
      loc = np.hstack((loc, loc_chunk + ind_ini_up))
      spike_fit = np.hstack((spike_fit, fit_chunk))
  return loc.astype(int), spike_fit