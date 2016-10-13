"""Locates local snippets by looking for local extrema.
"""
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


### AUXILIARY FUNCTIONS ###


def locate_snippets_segment(data, wl):
  """Locates snippets within a segment of data by finding local extrema.
  
  The absolute value of the selected points is larger than the absolute value
  of any other point within a window of width wl.

  Args:
    data: Segment of data.
    wl: Window length, must be odd.
    
  Returns:
    res: An array with the indices to the extrema.
  """
  if wl % 2 == 0:
    print "Window length should be odd"
    return -1
  half_window = (wl - 1) / 2
  # Add random noise in order to break equalities between local extrema
  # that attain the same values
  epsilon = 1e-8
  aux_data = np.absolute(data) + epsilon * np.random.rand(1, data.shape[0])
  local_extrema = np.ones((1, data.shape[0]))
  # Each point is compared to all the other points within a window of width
  # wl, local_extrema will be True at that point only if the
  # comparison is True for all the points
  for i in np.concatenate((range(-half_window, 0), range(1, half_window + 1))):
    local_extrema = (np.logical_and(local_extrema,
                                    aux_data > np.roll(aux_data, i)))
  res = np.nonzero(local_extrema)[1]
  return res


### FUNCTIONS ###


def locate_snippets(data, ind_ini, ind_end, wl):
  """Locates snippets by finding local extrema.
  
  The absolute value of the selected points is larger than the absolute value
  of any other point within a window of width wl. 

  Args:
    data: Segment of data.
    ind_ini: Indices at which segments to be processed start.
    ind_end: Indices at which segments to be processed end.
    wl: Window length, must be odd.
    
  Returns:
    An array with the indices to the extrema.
  """
  res = np.array([])
  for i in range(len(ind_ini)):
    aux_data = data[ind_ini[i]:ind_end[i]]
    aux_res = locate_snippets_segment(aux_data, wl)
    res = np.hstack((res,ind_ini[i] + aux_res))
  return res.astype(int)
