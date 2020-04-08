import numpy as np


def omp(A, b, k):
    """
    OMP Solve the P0 problem via OMP

    Solves the following problem:
      min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k

    The solution is returned in the vector x
    """
    assert k <= A.shape[1]

    x = np.zeros((np.shape(A)[1], 1))  # define x as a column vector
    support = []  # container for the chosen atoms (will store indices)

    b = b.reshape(-1, 1)
    r = b.copy()  # the residual. reshaped b to ensure it's a column vector
    coeff = None

    for i in range(k):
        error_vectors = r - (A.T @ r).T * A
        error_values = np.linalg.norm(error_vectors, ord=2, axis=0)
        support.append(np.argmin(error_values))

        A_s = A[:, support]
        coeff = np.linalg.inv(A_s.T @ A_s) @ A_s.T @ b
        r = b - A_s @ coeff

    x[support] = coeff
    return x.reshape(-1, 1)