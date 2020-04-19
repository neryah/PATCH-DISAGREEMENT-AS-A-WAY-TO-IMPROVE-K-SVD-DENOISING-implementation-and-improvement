clear; clc; close all;

%% Part A: Data Construction and Parameter-Setting

% Read an image
im = imread('barbara.png'); 


% Convert to double
im = double(im);

% Show the image
figure; imshow(im,[]); title('Original image');

% Patch dimensions [height, width]
patch_size = [8 8];

% Set the seed for the random generator
seed = 66;
 
% Set a fixed random seed to reproduce the results
rng(seed);

% Set the standard-deviation of the Gaussian noise
sigma = 20;

% Add noise to the input image
noise = sigma * randn(size(im));
noisy_im = im + noise;

% Compute the PSNR of the noisy image and print its value
psnr_noisy = compute_psnr(im, noisy_im);
fprintf('PSNR of the noisy image is %.3f\n\n', psnr_noisy);
 
% Show the original and noisy images
figure (1);
subplot(2,2,1); imshow(im,[]);
title('Original');
subplot(2,2,2); imshow(noisy_im,[]);
title(['Noisy, PSNR = ' num2str(psnr_noisy)]);

% Initialize the dictionary - unitary or not
% D_DCT = build_dct_unitary_dictionary(patch_size);
D_DCT = odctndict(patch_size(1), 256, 2);

% Set K - the cardinality of the solution.
% This will serve us later as the stopping criterion of the pursuit
K = 4;

% Set the number of iterations for K-SVD and for Patch-Disagreement
num_iters_ksvd = 20;
num_iters_disagreement = 30;

%% Part B: Running K-SVD and Patch-Disagreement

[im_res, D_res_ksvd] = k_svd(noisy_im, D_DCT, patch_size, K, num_iters_ksvd);

% Compute the PSNR of the resulting image and print its value
psnr_ksvd = compute_psnr(im, im_res);
fprintf('PSNR after K-SVD is %.3f\n\n', psnr_ksvd);


subplot(2,2,3); imshow(im_res,[]);
title(['After K-SVD Denoising, PSNR = ' num2str(psnr_ksvd)]);



% Resulting dictionary from K-SVD given as initial to Patch-Disgareement,
% as it done in paper
[im_res, D_res_disagreement] = disagreement(im, noisy_im, D_res_ksvd, patch_size, K, num_iters_disagreement);

% Compute the PSNR of the resulting image and print its value
psnr_disagreement = compute_psnr(im, im_res);
fprintf('PSNR after Patch-Disagreement is %.3f\n\n\n', psnr_disagreement);

fprintf('By Patch-Disagreement achieved improvement of %.3f in PSNR!\n', psnr_disagreement-psnr_ksvd);


subplot(2,2,4); imshow(im_res,[]);
title(['After Patch-Disagreement Denoising, PSNR = ' num2str(psnr_disagreement)]);



% Show the unitary DCT dictionary
figure(2);
subplot(1,3,1); show_dictionary(D_DCT);
title('Unitary DCT Dictionary');

% Show the learned by K-SVD dictionary
subplot(1,3,2); show_dictionary(D_res_ksvd);
title('Learned from K-SVD Dictionary');

% Show the learned by Patch-Disagreement
subplot(1,3,3); show_dictionary(D_res_disagreement);
title('Learned from Patch-Disagreement Dictionary');
 