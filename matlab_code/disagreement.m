function [result_im, D] = disagreement(noisy_im, D_init, patch_size, iters_num, sigma, gain, mu, alpha)
    %% Description
    % Patch-Disagreement Denoising implementation according to paper:
    % "PATCH-DISAGREEMENT AS A WAY TO IMPROVE K-SVD DENOISING"
    % by Elad and Romano.
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
    % alpha - new parameter we propose to vary in our paper

    %% Algorithm implementation
    % Divide the image into FULLY overlapping patches
    Y = im2col(noisy_im,patch_size,'sliding');
    
    [~,atoms_num] = size(D_init);
    
    % X holds sparse representations
    X = sparse(size(D_init,2), size(Y,2));
    
    % D holds a dictionary
    D = D_init;
    
    % Q holds disagreement patches as columns
    Q = sparse(size(Y,1),size(Y,2));
    
    % Holds the result
    result_im = zeros(size(noisy_im));
    
    % Target error for OMP
    epsilon = sqrt(prod(patch_size)) * sigma * gain;
    
    % Every iteration of algorithm does two iterations of Sparse Coding and
    % Dictionary Update, and one of Disagreement-Update
    bar = waitbar(0,'Running Patch-Disagreement...');
    for J = 1 : 2*iters_num
        %% Sparse coding stage
        curY = Y - Q;
        
        % Use Error-Constrained Batch OMP to compute the
        % sparse representation vectors x_i,
        % for every data sample y_i.
        X = omp2(D'*curY,sum(curY.*curY),D'*D, epsilon);

        %% Codebook update stage
        % Compute total error matrix now for faster computations.
        % It will be updated on fly to avoid massive recalculations.
        E = curY - D * X;
        
        [D, X, ~] = dictUpdate(atoms_num, D, X, E);
        
        %% Image reconstruction stage and weighed averaging with noisy
        current_local_estimate_patches = D * X;
        current_global_estimate = col_to_im(current_local_estimate_patches, patch_size, size(noisy_im));
        current_global_estimate = (1 - mu)*current_global_estimate + mu * noisy_im;
        result_im = current_global_estimate;
        
        %% Disagreement - update stage
        if mod(J,2)>0
            patches_from_restored_estimate = im2col(current_global_estimate, patch_size);
            Q = current_local_estimate_patches - patches_from_restored_estimate;
            Q = alpha * Q;
        else
            Q = 0;
        end
        
        waitbar(J/(2*iters_num), bar, 'Running Patch-Disagreement...');
        %fprintf('Ended iteration %i of Patch-Disagreement\n',J);
    end
    close(bar);
end