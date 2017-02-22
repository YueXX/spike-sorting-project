"""Test for segment_data.py.
"""
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
from segment_data import *
from preprocess_data import preprocess_data
import local_directories as ldir

def segment_data_plot_test(electrodes, wl, max_fraction, min_fraction):
  """ Test for segment_data.
  """
  for e in electrodes:
    data_file = ldir.DATA_PATH + str(e) + ".npy"
    data_electrode = np.load(data_file)
    data = preprocess_data(data_electrode)
    plt.ion()
    segment_data(data, wl, max_fraction, min_fraction, plot_res=True)
  

def main():
  plt.close("all")
  test_plot = True
  electrodes = [58]
  wl = 41
  max_fraction = 0.2
  min_fraction = 0.1
  if test_plot:
    segment_data_plot_test(electrodes, wl, max_fraction, min_fraction)


if __name__ == "__main__":
    main()
