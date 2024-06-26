import numpy as np

def get_eigen(A, max_iterations=10000, tolerance=1e-10):
  """
  Computes the eigenvalues and eigenvectors of a square matrix using the QR algorithm.

  Parameters:
  A (numpy.ndarray): The input square matrix for which to compute the eigenvalues and eigenvectors.
  max_iterations (int, optional): The maximum number of iterations to perform (default is 10000).
  tolerance (float, optional): The convergence tolerance (default is 1e-10).

  Returns:
  eigenvalues (numpy.ndarray): The eigenvalues of the matrix.
  eigenvectors (numpy.ndarray): The eigenvectors of the matrix.

  Raise:
  AssertionError: If the input matrix is not square.

  Notes:
  - The algorithms performs QR decomposition iteratively and checks for convergence
    by evaluating the norm of the off-diagonal elements.
  - Convergence is determined when the norm of the lower triangular part of the matrix
    is less than the specified tolerance.
  """
  m, n = A.shape
  assert m == n, "Matrix must be square"

  Q_total = np.eye(n)
  A_k = A.copy()

  for i in range(max_iterations):
    Q, R = qr_decomposition(A_k)
    A_k = R @ Q
    Q_total = Q_total @ Q

    off_diagonal_norm = np.sqrt(np.sum(np.tril(A_k, -1) ** 2))
    if off_diagonal_norm < tolerance:
      break

  eigenvalues = np.diag(A_k)
  eigenvectors = Q_total

  return eigenvalues, eigenvectors
