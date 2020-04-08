import numpy as np

def compute_stat(est_patches, orig_patches, est_coeffs):
    
    # COMPUTE_STAT Compute and print usefull statistics of the pursuit and
    # learning procedures
    #
    # Inputs:
    #  est_patches  - A matrix, containing the recovered patches as its columns
    #  orig_patches - A matrix, containing the original patches as its columns
    #  est_coeffs   - A matrix, containing the estimated representations, 
    #                 leading to est_patches, as its columns
    #
    # Outputs:
    #  residual_error  - Average Mean Squared Error (MSE) per pixel
    #  avg_cardinality - Average number of nonzeros that is used to represent 
    #                    each patch
    #

    # Compute the Mean Square Error per patch
    MSE_per_patch = np.sum((est_patches - orig_patches)**2,axis=0)

    # Compute the average 
    residual_error = np.mean(MSE_per_patch)/np.shape(orig_patches)[0]

    # Compute the average number of non-zeros
    avg_cardinality = np.sum(np.abs(est_coeffs) > 10**(-10)) / np.shape(est_coeffs)[1]

    # Display the results
    print('Residual error %2.2f, Average cardinality %2.2f' % (residual_error, avg_cardinality))
        
    return residual_error, avg_cardinality
