# spike-sorting

Code to perform spike sorting on a multi-electrode array using one of three methods:

- **Single-electrode sorting via clustering**: Clusters snippets of data with high activity to produce spike and waveform estimates for a single electrode. 

- **Optimization-based single-electrode sorting**: Estimates the spike locations and the corresponding waveforms for a single electrode by optimizing the fit to the data.

- **Optimization-based multi-electrode sorting**: Estimates the spike locations for each cell over the whole array, as well as the corresponding waveforms, by optimizing the fit to the data.

## 1. Single-electrode sorting via clustering

Locates snippets of data with high activity, upsamples them and clusters them.

### Snippet extraction

The data is segmented into snippets, which are then upsampled and aligned. 

#### *segment_data.py*

Segments data by thresholding the energy over a windows of a certain length. The threshold is set so that a certain fraction of the data is selected.

- *segment_data(data, wl, max_fraction, min_fraction)*: A point *i* is selected if the energy within a window of length *wl* centered at *i* is above a certain threshold. The threshold is set so that the fraction of data that is selected is between *min_fraction* and *max_fraction*

#### *locate_snippets.py*

Locates snippets by finding local extrema in the data

- *locate_snippets(data, ind_ini, ind_end, wl)*: Locates points that have the largest magnitude within within a window of length *wl*. The search is restricted to the segments indexed by *ind_ini* and *ind_end*.

#### *upsample_snippets.py*

Upsamples snippets and centers them. 

- *upsample_snippets(data, snippet_indices, wl, uf)*: Snippets located at *snippet_indices* are separated into two types: those that first have a maximum
and then a minimum (corresponding to soma-dendritic spikes) and vice versa (axonal spikes). The snippets are upsampled by a factor *uf* and then aligned so that their minimum coincides.

### K-means clustering

#### *cluster_snippets.py*:

- *cluster_snippets(snippets, locations, k, data, uf, wl, p=0, average=True, save_file="")*

Clusters snippets produced by upsample_snippets using k-means, optionally after projecting onto the first p principal components.

## Additional functions:

### Data preprocessing

#### *preprocess_data.py*

Applies a high-pass filter and normalizes the amplitude of the electrode data.

- *filter_data(data, params)*: Applies a high-pass Butterworth filter. 

- *preprocess_data(data, params=None)*: Calls filter_data and then normalizes the data.

### Evaluation

## 2. Multi-electrode sorting 

Detect pattern of multi-electrode patterns, use non-negative matrix factorization

![Alt text](mysimsworldmap-v6.png?raw=true "Optional Title")





















