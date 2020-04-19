function [best_result_im, D] = k_svd(noisy_im, D_init, patch_size, T, iters_num)
    %% Description
    % K-SVD Denoising implementation according to paper:
    % "K-SVD: DESIGN OF DICTIONARIES FOR SPARSE REPRESENTATION"
    % by Elad, Aharon and Bruckstein.
    % This is not optimized and not ideal implementation.
    % Done as part of course project for course:
    % 236862 Sparse and Redundant Representations
    % Technion - IIT, 2020
    
    %% Parameters:
    % noisy_im - Noisy image to be denoised.
    % D_init - Initial dictionary, i.e. D_0.
    % patch_size - Size of patches to take from image.
    % T - cardinality of desired solution for OMP.
    % iters_num - Number of iterations to run.

    %% Algorithm implementation
    
    % Divide the image into fully overlapping patches
    Y = im2col(noisy_im,patch_size,'sliding');
    
    [~,atoms_num] = size(D_init);
    
    % X holds sparse representations
    X = zeros(size(Y));
    
    % D holds a dictionary
    D = D_init;
    
    for J = 1 : iters_num
        %% Sparse coding stage
        % Use Batch OMP to compute the sparse representation vectors x_i,
        % for every data sample y_i.

        X = omp(D'*Y,D'*D,T);

        %% Codebook update stage
        % Compute total error matrix now for faster computations.
        % It will be updated on fly to avoid massive recalculations.
        E = Y - D * X;
        
        for k = 1 : atoms_num
            % Define w_k as a group of examples which used atom d_k in
            % their sparse representation.
            w_k = find(X(k,:));
            
            % If w_k is empty there's no need to update this atom,
            % it was not used.
            if size(w_k) ~= 0
                % Restrict E_k by choosing only the columns corresponding
                % those elements that initially used d_k in their representation.
                E_kr = E(:,w_k) + D(:,k) * X(k,w_k);
                
                [U, Sigma, V] = svds(E_kr,1);
                D(:,k) = U(:,1);
                X(k,w_k) = Sigma(1,1) * V(:,1)';
                
                % Update E on fly
                E(:,w_k) = E_kr - D(:,k) * X(k,w_k);
            end
        end
        
        % fprintf('Ended iteration %i of K-SVD\n',J);
    end
    
    % Image reconstruction
    current_global_estimate_patches = D * X;
    best_result_im = col_to_im(current_global_estimate_patches, patch_size, size(noisy_im));
end

