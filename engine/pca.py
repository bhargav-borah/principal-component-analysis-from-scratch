import pandas as pd
import numpy as np

from covariance import covariance
from get_covariance_matrix import get_covariance_matrix
from get_eigen import get_eigen
from get_mean_values import get_mean_values
from qr_decomposition import qr_decomposition

def pca(X , n_components):
  """
  Perform Principal Component Analysis (PCA) on the dataset.

  Args:
  X : array-like, shape (n_samples, n_features)
      The data matrix, where n_samples is the number of samples and n_features is the number of features.
  n_components : int
      The number of principal components to return.

  Returns:
  X_reduced: array-like, shape (n_samples, n_components)
      The transformed data matrix with reduced dimensions.

  Notes:
  - The input data matrix `X` should be standardized (i.e., rescaled to have mean zero and variance one) if the features are on different scales.
  - Each column of the returned eigenvector matrix corresponds to an eigenvector.
  """
  X_centered = X - np.mean(X, axis=0)
  cov_mat = get_covariance_matrix(X_centered)

  eigenvalues, eigenvectors = get_eigen(cov_mat)

  # Note: Each column of `eigenvectors` corresponds to an eigenvector.

  sorted_indices = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sorted_indices]
  eigenvectors = eigenvectors[:, sorted_indices]
  eigenvectors = eigenvectors[:, :n_components]

  X_reduced = np.dot(X_centered, eigenvectors)

  return X_reduced
