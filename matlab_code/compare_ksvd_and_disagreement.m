function [psnr_ksvd, psnr_disagreement] = ...
    compare_ksvd_and_disagreement(im, noisy_im, D_init, patch_size, K, num_iters_ksvd, num_iters_disagreement)

    [im_res, D_res_ksvd] = k_svd(noisy_im, D_init, patch_size, K, num_iters_ksvd);

    % Compute the PSNR of the resulting image and print its value
    psnr_ksvd = compute_psnr(im, im_res);
    
    % Resulting dictionary from K-SVD given as initial to Patch-Disgareement,
    % as it done in paper
    [im_res, ~] = disagreement(im, noisy_im, D_res_ksvd, patch_size, K, num_iters_disagreement);

    % Compute the PSNR of the resulting image and print its value
    psnr_disagreement = compute_psnr(im, im_res);

end


 