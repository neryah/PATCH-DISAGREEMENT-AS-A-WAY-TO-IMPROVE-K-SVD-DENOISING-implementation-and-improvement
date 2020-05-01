function [psnr_ksvd, psnr_disagreement] = ...
    compare_ksvd_and_disagreement(im, noisy_im, D_init, patch_size, sigma, gain, num_iters_ksvd, num_iters_disagreement, mu, alpha)
    %% Description
    % Given a clear and noisy images, run both  K-SVD and "Sharing the disagreement"
    % algorithms and return their denoising results as PSNR.
    
    % Done as part of course project for course:
    % 236862 Sparse and Redundant Representations
    % Technion - IIT, 2020
    %%
    
    % Denoise using K-SVD
    [im_res, D_res_ksvd] = k_svd(noisy_im, D_init, patch_size, num_iters_ksvd, sigma, 1, mu);

    % Compute the PSNR of the resulting image
    psnr_ksvd = compute_psnr(im, im_res);
    
    % Resulting dictionary from K-SVD given as initial to Patch-Disgareement,
    % as it done in paper
    [im_res, ~] = disagreement(noisy_im, D_res_ksvd, patch_size, num_iters_disagreement, sigma, gain, mu, alpha);

    % Compute the PSNR of the resulting image
    psnr_disagreement = compute_psnr(im, im_res);

end
