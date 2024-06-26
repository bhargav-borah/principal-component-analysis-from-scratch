import pandas as pd
import numpy as np

from covariance import covariance

def get_covariance_matrix(df: pd.DataFrame):
  """
  Calculate the covariance matrix for the given DataFrame.

  Parameters:
  df (pd.DataFrame): A pandas DataFrame where each column represents a feature.

  Returns:
  np.ndarray: A 2D numpy array representing the covariance matrix of the features in the DataFrame.
  """
  N = len(df.columns)

  cov_matrix = np.zeros((N, N))

  for i, feature_1 in enumerate(df.columns):
    for j, feature_2 in enumerate(df.columns):
      cov_matrix[i, j] = covariance(df[feature_1], df[feature_2])

  return cov_matrix
