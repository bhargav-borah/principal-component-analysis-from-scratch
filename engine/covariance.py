import pandas as pd

def covariance(feature_1: pd.Series, feature_2: pd.Series):
  """
  Calculate the covariance between two features.

  Parameters:
  feature_1 (pd.Series): The first feature as a pandas Series.
  feature_2 (pd.Series): The second featuere as a pandas Series.

  Returns:
  float: The covariance between feature_1 and feature_2.
  """
  mean_1 = feature_1.mean()
  mean_2 = feature_2.mean()

  N = len(feature_1)

  sum = 0

  for i in range(N):
    sum += (feature_1[i] - mean_1) * (feature_2[i] - mean_2)

  return sum / (N - 1)
