import pandas as pd

def get_mean_values(df: pd.DataFrame):
  """
  Calculate the mean values of each column in the DataFrame.

  Parameters:
  df (pd.DataFrame): The input DataFrame for which the mean values of each column are to be calculated.

  Returns:
  mean_values: A list containing the mean value of each column in the same order as the columns in the DataFrame.
  """
  mean_values = []

  for feature in df.columns:
    mean_values.append(df[feature].mean())

  return mean_values
