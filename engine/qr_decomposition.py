import numpy as np

def qr_decomposition(matrix: np.ndarray):
  """
  Perform QR decomposition of a given matrix using the Gram-Schmidt process.

  Args:
    matrix (np.ndarray): The matrix which has to undergo QR decomposition.

  Returns:
    Q (np.array): The orthogonal matrix Q of the decomposition.
    R (np.array): The upper triangular R of the decomposition.
  """
  mat = np.copy(matrix)
  num_cols = mat.shape[1]

  v_vectors = []
  u_vectors = []

  for i in range(num_cols):
    v_vectors.append(mat[:, i])

  for i in range(num_cols):
    if i == 0:
      u_vectors.append(v_vectors[0] / np.linalg.norm(v_vectors[0]))
    else:
      u_vector = v_vectors[i]
      for idx in range(i - 1, -1, -1):
        u_vector -= np.dot(u_vectors[idx], v_vectors[i]) * u_vectors[idx]
      u_vector /= np.linalg.norm(u_vector)
      u_vectors.append(u_vector)

  Q = np.array(u_vectors).T
  R = np.dot(Q.T, matrix)

  return Q, R
