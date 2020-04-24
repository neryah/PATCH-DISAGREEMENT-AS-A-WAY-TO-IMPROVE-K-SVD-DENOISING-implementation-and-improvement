function [best_result_im, D] = disagreement(orig_im, noisy_im, D_init, patch_size, T, iters_num)
    %% Description
    % Patch-Disagreement Denoising implementation according to paper:
    % "PATCH-DISAGREEMENT AS A WAY TO IMPROVE K-SVD DENOISING"
    % by Elad and Romano.
    % This is not optimized and not ideal implementation.
    % Done as part of course project for course:
    % 236862 Sparse and Redundant Representations
    % Technion - IIT, 2020
    
    %% Parameters:
    % orig_im - Original image. Given for comparison of PSNR only.
    % noisy_im - Noisy image to be denoised.
    % D_init - Initial dictionary, i.e. D_0.
    % patch_size - Size of patches to take from image.
    % T - cardinality of desired solution for OMP.
    % iters_num - Number of iterations to run.

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
    
    % Hold the best result between iterations
    best_psnr = 0;
    best_result_im = zeros(size(noisy_im));
    
    for J = 1 : 2*iters_num
        %% Sparse coding stage
        curY = Y - Q; % minus disagreement!
        % Use Batch OMP to compute the sparse representation vectors x_i,
        % for every data sample y_i.

        X = omp(D'*curY,D'*D,T);

%         for patch_num = 1 : patches_num
%             X(:,patch_num) = omp(D, Y(:,patch_num) - Q(:,patch_num), T);
%         end

        %% Codebook update stage
        % Compute total error matrix now for faster computations.
        % It will be updated on fly to avoid massive recalculations.
        E = curY - D * X;
        
        [D, X, ~] = dictUpdate(atoms_num, D, X, E);
        

        %% Image reconstruction stage
        current_local_estimate_patches = D * X;
        current_global_estimate = col_to_im(current_local_estimate_patches, patch_size, size(noisy_im));
        current_global_estimate = 0.995*current_global_estimate + 0.005*(noisy_im);
        current_psnr = compute_psnr(orig_im, current_global_estimate);
        if current_psnr > best_psnr
           best_psnr = current_psnr;
           best_result_im = current_global_estimate;
           fprintf('Found better PSNR at iteration %i\n',floor(J/2));
        end
        
        % fprintf('PSNR at iteration %i: %4.4f\n',J,current_psnr);
        
        %% Disagreement - update stage
        patches_from_restored_estimate = im2col(current_global_estimate, patch_size);
        if mod(J,2)>0
            Q=(current_local_estimate_patches - patches_from_restored_estimate);
        else
            fprintf('Norm of Q is %4.4f\n',norm(Q,'fro'));
            fprintf('PSNR at iteration %i: %4.4f\n',J/2,current_psnr);
            Q = 0;
        end
        % fprintf('Norm of Q is %4.4f\n',norm(Q,'fro'));
        % fprintf('Ended iteration %i of Patch-Disagreement\n',J);
    end
   
end

