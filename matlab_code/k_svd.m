function [result_im, D] = k_svd(noisy_im, D_init, patch_size, iters_num, sigma, gain, mu)
    %% Description
    % K-SVD Denoising implementation according to paper:
    % "K-SVD: DESIGN OF DICTIONARIES FOR SPARSE REPRESENTATION"
    % by Elad, Aharon and Bruckstein.
    %
    % Done as part of course project for course:
    % 236862 Sparse and Redundant Representations
    % Technion - IIT, 2020
    
    %% Parameters:
    % noisy_im - Noisy image to be denoised.
    % D_init - Initial dictionary, i.e. D_0.
    % patch_size - Size of patches to take from image.
    % iters_num - Number of iterations to run.
    % sigma - Noise level for Error-Constrained OMP
    % gain - Noise gain for Error-Constrained OMP
    % mu - Relative part of noisy image for averaging

    %% Initialization
    
    % Divide the image into fully overlapping patches
    Y = im2col(noisy_im,patch_size,'sliding');
    
    [~,atoms_num] = size(D_init);
    
    % X holds sparse representations
    X = zeros(size(Y));
    
    % D holds a dictionary
    D = D_init;
    
    % Target error for OMP
    epsilon = sqrt(prod(patch_size)) * sigma * gain;
    
    %% Run the algorithm
    
    bar = waitbar(0,'Running K-SVD...');
    for J = 1 : iters_num
        %% Sparse coding stage
        % Use Batch OMP to compute the sparse representation vectors x_i,
        % for every data sample y_i.

        X = omp2(D'*Y,sum(Y.*Y),D'*D, epsilon);

        %% Codebook update stage
        % Compute total error matrix now for faster computations.
        % It will be updated on fly to avoid massive recalculations.
        E = Y - D * X;
        
        [D, X, ~] = dictUpdate(atoms_num, D, X, E);
        
        waitbar(J/iters_num, bar, 'Running K-SVD...');
        %fprintf('Ended iteration %i of K-SVD\n',J);
    end
    
    %% Image reconstruction and weighed averaging with noisy
    current_global_estimate_patches = D * X;
    result_im = col_to_im(current_global_estimate_patches, patch_size, size(noisy_im));
    result_im = (1 - mu) * result_im + mu * (noisy_im);
    
    close(bar);
end

