"""Test for locate_snippets.
"""
#from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
from segment_data import *
from locate_snippets import *
from preprocess_data import preprocess_data
import local_directories as ldir
import numpy.testing as npt

def locate_snippets_test_simulated():
  """Tests locate_snippets using simulated data.
  """
  print "Testing locate_snippets_segment on simulated data"
  sim_data = np.array([1.2, 3, 4.1, -1.1, -2.2, -3.1, -1, 0.2, 5, 3, 4, 5.1])
  wl_1 = 3
  expected_extrema_1 = np.array([2, 5, 8, 11])
  extrema_1 = locate_snippets_segment(sim_data, wl_1)
  npt.assert_allclose(expected_extrema_1, extrema_1, 
                      err_msg="Test 1 failed")
  wl_2 = 7
  expected_extrema_2 = np.array([11])
  extrema_2 = locate_snippets_segment(sim_data, wl_2)
  npt.assert_allclose(expected_extrema_2, extrema_2, 
                      err_msg="Test 2 failed")
  ind_ini = [3, 6]
  ind_end = [5, 10]
  expected_extrema_3 = np.array([4, 8])
  extrema_3 = locate_snippets(sim_data, ind_ini, ind_end, wl_1)
  npt.assert_allclose(expected_extrema_3, extrema_3, 
                      err_msg="Test 3 failed")
  print "Tests completed"          
                      

def locate_snippets_test_real(electrodes):
  """Tests locate_snippets on real data.
  """
  wl = 41
  prefix_size = 6000
  max_fraction = 0.2
  min_fraction = 0.15
  plt.ion()
  for e in electrodes:
    data_electrode = np.load(ldir.DATA_PATH + str(e) + ".npy")
    all_data = preprocess_data(data_electrode)
    data = all_data[1:prefix_size]
    ind_ini, ind_end = segment_data(data, wl, max_fraction, min_fraction)
    extrema = locate_snippets(data, ind_ini, ind_end, wl)
    plt.figure(figsize=(35.0, 15.0))
    plt.plot(data, "--ob", markerfacecolor="None", label="Data")
    for i in range(len(ind_ini)):
      if i==0:
        leg_label="Segments"
      else:
        leg_label=None
      plt.plot(range(ind_ini[i], ind_end[i]),
                 data[ind_ini[i]:ind_end[i]], "--r", markerfacecolor="None",
                 label=leg_label)
    plt.plot(extrema, data[extrema], "og", label="Extrema")
    plt.legend()
    plt.title("Electrode " + str(e))
    

def main():
  plt.close("all")
  locate_snippets_test_simulated()
  electrodes = [123]
  locate_snippets_test_real(electrodes)
  # ipshell = InteractiveShellEmbed()
  # ipshell()


if __name__ == "__main__":
  main()