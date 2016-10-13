import os.path
import numpy as np
import h5py
from preprocess_data import preprocess_data

DEST_DIR = "/Users/cfgranda/Google Drive/spike_sorting/data/"
#DEST_DIR = "/usr/local/google/home/cfgranda/Google Drive/spike_sorting/data/"
# DATA_FILE = "/Users/cfgranda/Google Drive/Raw MEA Data/Spikes_all_channels.mat"
DATA_FILE = "/usr/local/google/home/cfgranda/Spikes_all_channels.mat"


def store_data_from_mat():
  """ This function converts electrode date in matlab format to a numpy array
  It is used once for each dataset."""
  aux_dictionary = h5py.File(DATA_FILE)
  all_data = aux_dictionary["allElecData"]
  for electrode in range(all_data.shape[1]):
    print "Electrode " + str(electrode)
    data_electrode = all_data[:, electrode].T
    # Store data
    save_file = (DEST_DIR + "electrode" + str(electrode))
    np.save(save_file, data_electrode)


def store_preprocessed_data():
  e_list = range(512)  
  for electrode in e_list:
    print "Electrode " + str(electrode)
    load_file = (DEST_DIR + "electrode" + str(electrode) + ".npy")
    save_file = (DEST_DIR + "electrode" + str(electrode) + "_prep")
    data_electrode = np.load(load_file)
    data = preprocess_data(data_electrode)
    np.save(save_file, data)
    

def main():   
  store_preprocessed_data()


if __name__ == "__main__":
  main()
