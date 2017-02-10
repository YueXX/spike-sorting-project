"""Upsamples snippets and centers them.

Snippets are separated into two types: those that first have a maximum
and then a minimum (corresponding to soma-dendritic spikes) and vice versa
(corresponding to axonal spikes). 
"""
import os.path
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def upsample_snippets(data, snippet_indices, wl_down, uf, savepath=None):
  """Upsamples snippets and centers them.

  Upsamples snippets, locates maximum and minimum within a window of width
  wl_down and centers the snippet around the second local extremum.
  Snippets are separated into two types: those that first have a maximum
  and then a minimum and vice versa. For usage see test function.

  Args:
    data: Array of data.
    snippet_indices: Indices at which there is a local extremum.
    wl_down: Window length on coarse grid, must be odd.
    uf: Upsampling factor.
    savepath: Path to save results.

  Returns:
    res: Dictionary containing:
         -maxmin_snips: Snippets with a maximum and then a minimum.
         -maxmin_loc: Locations of the snippets in maxmin_snips. 
         -minmax_snips: Snippets with a minimum and then a maximum.
         -minmax_loc: Locations of the snippets in minmax_snips.
  """
  n_samples = data.shape[0]
  n_snippets = snippet_indices.shape[0]
  h_down = (wl_down - 1) / 2
  assert wl_down % 2 == 1, ("Window length should be of the form 2 * h + 1," 
                            "where h is even.")
  assert h_down % 2 == 0, ("Window length should be of the form 2 * h + 1," 
                            "where h is even.")
  h_up = h_down * uf
  wl_up = 2 * h_up + 1

  # Upsample using a segment of larger width so that we can recenter
  indices = np.arange(-2 * h_down, 2.5 * h_down + 1)
  step = 1. / uf
  indices_fine = np.arange(-2 * h_down, 2.5 * h_down + step, step)
  # arange produces a small floating point error, which may cause an exception
  # while upsampling. Therefore we overwrite the first and last elements.
  indices_fine[0] = -2 * h_down
  indices_fine[-1] = 2.5 * h_down

  # Matrix to store upsampled snippets
  upsampled_matrix = np.zeros((n_snippets, wl_up))
  # Array to store location of upsampled snippets
  locations = np.zeros(n_snippets)
  # Arrays to indicate whether the snippet has a maximum and then a minimum
  # or vice versa
  maxmin = np.zeros(n_snippets)
  minmax = np.zeros(n_snippets)
  for i in range(0, n_snippets):
    ind = snippet_indices[i]
    # Ignore indices at the beginning and at the end of the data array.
    if ind < 1 * wl_down or ind > n_samples - 1.5 * wl_down:
      continue
    ind_snippet = range((ind - 2 * h_down),
                        (ind + 2 * h_down + h_down / 2 + 1))
    ext_snippet = data[ind_snippet]
    f = interp1d(indices, ext_snippet, kind="cubic")
    upsamp_snippet = f(indices_fine)

    # Locate the maximum and minimum within a window of width
    # wl_down of the original extremum
    central_ind = range(h_up, 3 * h_up + 1)
    arg_max = np.argmax(upsamp_snippet[central_ind])
    arg_min = np.argmax(-upsamp_snippet[central_ind])
    if arg_max < arg_min:
      center = h_up + arg_min
      maxmin[i] = 1
    else:
      # In case there is a minimum first, align it to lie at 0.25xwl_down
      center = h_up + arg_min + h_up / 2
      minmax[i] = 1
    upsamp_indices = range(center - h_up,
                           center + h_up + 1)
    upsampled_matrix[i, :] = upsamp_snippet[upsamp_indices]
    locations[i] = ind * uf + (center - 2 * h_up) 
  locations = locations.astype(int)
  maxmin_ind = np.nonzero(maxmin)[0]
  maxmin_snips = upsampled_matrix[maxmin_ind, :]
  maxmin_loc = locations[maxmin_ind]
  minmax_ind = np.nonzero(minmax)[0]
  minmax_snips = upsampled_matrix[minmax_ind, :]
  minmax_loc = locations[minmax_ind]
  res = {"maxmin_snips": maxmin_snips, "maxmin_loc": maxmin_loc, 
         "minmax_snips": minmax_snips, "minmax_loc": minmax_loc}
  if savepath is not None:
    np.save(savepath, res)
  return res