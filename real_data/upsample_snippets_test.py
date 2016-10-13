"""Test for upsample_snippets. 
"""
from IPython.terminal.embed import InteractiveShellEmbed
import matplotlib.pyplot as plt
import numpy as np
from upsample_snippets import upsample_snippets
from evaluate_fit import snippet_indices_plot

def upsample_snippets_test(uf):
  """Test for upsample_snippets.
  """
  snippet_max_min_1 = np.array([0.1, 0.4, 0.3, 0.2, -0.1, -0.05, -0.3, -0.05, 
                                0.02])
  snippet_max_min_2 = np.array([0.2, 0.3, 0.5, 1.2, -0.6, -0.8, -0.5, -0.2, 
                                -0.01])
  snippet_min_max_1 = np.array([-0.01, -0.7, -0.5, -0.1, 0.02, 0.4, 0.1, 0.2, 
                                0.1])
  aux_data = [0.03, -0.09, -0.03, 0.06, 0.03, -0.09, -0.03, 0.06,0.03, -0.09, 
              -0.03, 0.06]
  data = np.hstack((aux_data, snippet_max_min_1, aux_data, snippet_min_max_1, 
                   aux_data, snippet_max_min_2, aux_data))
  wl_down = len(snippet_max_min_1)
  wl_up = (wl_down - 1) * uf + 1
  center = (wl_down - 1) / 2 
  n_data = len(aux_data)
  snippet_indices = np.array([n_data + center, 2 * n_data + wl_down + center,
                              3 * n_data + 2 * wl_down + center])
  res = upsample_snippets(data, snippet_indices, wl_down, uf)
  for i in range(res["maxmin_snips"].shape[0]):
    (data_ind, waveform_ind, ind_coarse, 
     ind_fine) = snippet_indices_plot(res["maxmin_loc"][i], uf, wl_up)
    snippet = res["maxmin_snips"][i,:]
    data_snippet = data[data_ind]
    plt.figure(figsize=(35.0, 15.0))
    plt.plot(ind_coarse, data_snippet, "ob", markerfacecolor="None", 
             label="Data")
    plt.plot(ind_fine, snippet, "--xb", label="Interpolated snippet")
    plt.title("Upsampling factor: " + str(uf))
    plt.legend
  for i in range(res["minmax_snips"].shape[0]):
    (data_ind, waveform_ind, ind_coarse, 
     ind_fine) = snippet_indices_plot(res["minmax_loc"][i], uf, wl_up)
    snippet = res["minmax_snips"][i, :]
    data_snippet = data[data_ind]
    plt.figure(figsize=(35.0, 15.0))
    plt.plot(ind_coarse, data_snippet, "ob", markerfacecolor="None", 
             label="Data")
    plt.plot(ind_fine, snippet, "--xr", label="Interpolated snippet")
    plt.title("Upsampling factor: " + str(uf))
    plt.legend()
    

def main():
  plt.close("all")
  uf_val = [2, 5, 10]
  for uf in uf_val:
    upsample_snippets_test(uf)


if __name__ == "__main__":
    main()