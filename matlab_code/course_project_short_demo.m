%% Description
% A short demo demonstrating the result of comparison of K-SVD algorithm
% and Algorithm 1 i.e. "Sharing the Disagreement" from paper:
% "PATCH-DISAGREEMENT AS A WAY TO IMPROVE K-SVD DENOISING"
% by Elad and Romano.
% Requires ksvdbox13 if the initial DCT dictionary is not required
% to be unitary, requires also ompbox10 to use Error-Constrained OMP.
% Alpha is an additional parameter we propose for parametrization. For
% original implementation of Algorithm 1, this value equals 1.
%
% Done as part of course project for course:
% 236862 Sparse and Redundant Representations
% Technion - IIT, 2020

%% Part A: Data Construction and Parameter-Setting
clear; clc; close all;

% Read an image
im = imread('barbara.png'); 

% Convert to double
im = double(im);

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
 
% Initialize the dictionary - unitary or not
% D_DCT = build_dct_unitary_dictionary(patch_size);
D_DCT = odctndict(patch_size(1), 256, 2);

% Set gain as parameter for K-SVD / Patch-Disagreement
gain_ksvd = 1.0;
gain_disagreement = 1.16;

% Set proposed parameter alpha variations
alpha = 1;

% Set the number of iterations for K-SVD and for Patch-Disagreement
num_iters_ksvd = 20;
num_iters_disagreement = 30;

% Set weight for noisy image for averaging
mu = 0.005;

%% Part B: Running K-SVD and Patch-Disagreement

[im_res_ksvd, D_res_ksvd] = k_svd(noisy_im, D_DCT, patch_size, num_iters_ksvd, sigma, gain_ksvd, mu);

% Compute the PSNR of the resulting image and print its value
psnr_ksvd = compute_psnr(im, im_res_ksvd);
fprintf('PSNR after K-SVD is %.3f\n\n', psnr_ksvd);


% Resulting dictionary from K-SVD given as initial to Patch-Disgareement,
% as it done in paper
[im_res_disagreement, D_res_disagreement] = disagreement(noisy_im, D_res_ksvd, patch_size, num_iters_disagreement, sigma, gain_disagreement, mu, alpha);

% Compute the PSNR of the resulting image and print its value
psnr_disagreement = compute_psnr(im, im_res_disagreement);
fprintf('PSNR after Patch-Disagreement is %.3f\n\n', psnr_disagreement);

fprintf('By Patch-Disagreement achieved improvement of %.3f in PSNR!\n', psnr_disagreement-psnr_ksvd);



%% Part C: Show results

% Show the image
figure; imshow(im,[]); title('Original image');

figure (1);
subplot(2,2,1); imshow(im,[]);
title('Original');

subplot(2,2,2); imshow(noisy_im,[]);
title(['Noisy, PSNR = ' num2str(psnr_noisy)]);

subplot(2,2,3); imshow(im_res_ksvd,[]);
title(['After K-SVD Denoising, PSNR = ' num2str(psnr_ksvd)]);

subplot(2,2,4); imshow(im_res_disagreement,[]);
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
 