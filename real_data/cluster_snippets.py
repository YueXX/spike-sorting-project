"""Clusters snippets.

Snippets are clustered applying k means after an optional projection
onto the first p principal components.
"""
# from IPython.terminal.embed import InteractiveShellEmbed
import numpy as np
from scipy.cluster.vq import *


def cluster_snippets(snippets, locations, data, k, PCA, n_pc, average):
  """Clusters snippets applying k means.

  Clusters snippets applying k means. Optionally, the data are projected onto 
  their principal components before clustering.

  Args:
    snippets: Matrix of snippets; each snippet is a row.
    locations: Location of the snippet on the upsampled grid.
    data: Data.
    k: Parameter for k means.
    PCA: If True we project onto n_pc principal components before clustering.
    n_pc : Number of principal components.
    average: Determines whether the method returns the low-dimensional k-means 
             (projected onto the original space) (average=False) or averages 
             the snippets in each cluster (average=True).

  Returns:
    res: Dictionary containing:
         -wav: List of waveforms.
         -loc: Locations of the snippets corresponding to each waveform.
  """
  # Projection onto principal components
  if PCA:
    aux_mean = np.mean(snippets, axis = 0)
    centered_snips = snippets - aux_mean
    U_mat, singval, singvec = np.linalg.svd(centered_snips,
                                            full_matrices=False)
    pc = singvec[0:n_pc, :]
    points = np.dot(centered_snips, pc.T)
  else:
    points = snippets
  centroids, assignments = kmeans2(points, k, minit="points")
  loc = []
  ind_wav = []
  for i in range(k):
    nnz_ind = np.nonzero(assignments == i)[0]
    ind_wav.append(nnz_ind)
    loc.append(locations[nnz_ind])
  if PCA:
    if average:
      # Average each cluster
      wav = np.zeros((k, snippets.shape[1]))
      for i in range(0, k):
        wav[i, :] = np.mean(snippets[ind_wav[i], :], axis=0)
    else:
      # Project back
      wav = aux_mean + np.dot(centroids, pc)
  else:
    # No projection
    wav = centroids
  res = {"wav": wav, "loc": loc}
  return res