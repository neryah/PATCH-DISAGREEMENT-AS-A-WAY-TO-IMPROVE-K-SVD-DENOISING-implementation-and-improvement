import numpy as np
from compute_stat import compute_stat
from batch_thresholding import batch_thresholding
from numpy.linalg import svd


def unitary_dictionary_learning(Y, D_init, num_iterations, pursuit_param):
    # UNITARY_DICTIONARY_LEARNING Train a unitary dictionary via
    # Procrustes analysis.
    #
    # Inputs:
    #   Y              - A matrix that contains the training patches
    #                    (as vectors) as its columns
    #   D_init         - Initial UNITARY dictionary
    #   num_iterations - Number of dictionary updates
    #   pursuit_param  - The stopping criterion for the pursuit algorithm
    #
    # Outputs:
    #   D          - The trained UNITARY dictionary
    #   mean_error - A vector, containing the average representation error,
    #                computed per iteration and averaged over the total
    #                training examples
    #   mean_cardinality - A vector, containing the average number of nonzeros,
    #                      computed per iteration and averaged over the total
    #                      training examples

    # Allocate a vector that stores the average representation error per iteration
    mean_error = np.zeros(num_iterations)

    # Allocate a vector that stores the average cardinality per iteration
    mean_cardinality = np.zeros(num_iterations)

    D = D_init

    # Run the Procrustes analysis algorithm for num_iterations
    for i in range(num_iterations):
        # Compute the representation of each noisy patch
        [X, A] = batch_thresholding(D, Y, pursuit_param)

        # Compute and display the statistics
        print('Iter %2d: ' % i)
        mean_error[i], mean_cardinality[i] = compute_stat(X, Y, A)

        # Solve D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
        # where 'A' is a matrix that contains all the estimated coefficients,
        # and 'Y' contains the training examples. Use the Procrustes algorithm.
        # Write your code here... D = ???
        U, S, Vt = svd(Y @ A.T)
        D = U @ Vt

    return D, mean_error, mean_cardinality
