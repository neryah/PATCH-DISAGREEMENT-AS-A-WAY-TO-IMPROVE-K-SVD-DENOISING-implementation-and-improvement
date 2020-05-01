%% Description
% This script demonstrates the result of proposed paramterization of
% "Sharing the Disagreement" algorithm from paper:
% "PATCH-DISAGREEMENT AS A WAY TO IMPROVE K-SVD DENOISING"
% by Elad and Romano.
%
% Requires ksvdbox13 if the initial DCT dictionary is not required
% to be unitary, requires also ompbox10 to use Error-Constrained OMP.
%
% Alpha is an additional parameter we propose for parametrization. For
% original implementation of Algorithm 1, this value equals 1.
% For all testing images from original paper, we do K-SVD denoising,
% and compare the result to result of Algorithm 1 where after
% Disagreement-Update step we multiply resulting disagreement patches
% by given parameter alpha.
% Hence, in original paper this parameter always equals to 1.
% Here we iterate on values between 0 to 2 with step of 0.1.
% After checking all images we do an average and present a graph of
% improvement in PSNR compared to K-SVD agains value of alpha.
%
% Done as part of course project for course:
% 236862 Sparse and Redundant Representations
% Technion - IIT, 2020

%% Part A: Data Construction and Parameter-Setting
clear; clc; close all;

% Set a fixed testing images set as in paper
images = ["barbara.png","boat.png","couple.bmp","fingerprint.png","house.png","peppers.png"];


% Patch dimensions [height, width]
patch_size = [8 8];

% Set the seed for the random generator
seed = 66;
 
% Set a fixed random seed to reproduce the results
rng(seed);

% Set the standard-deviation of the Gaussian noise
sigma = 20;

% Initialize the dictionary - unitary or not
% D_DCT = build_dct_unitary_dictionary(patch_size);
D_DCT = odctndict(patch_size(1), 256, 2);

% Set gain as parameter for K-SVD / Patch-Disagreement
gain_ksvd = 1.0;
gain_disagreement = 1.16;

% Set proposed parameter alpha variations
alphas = linspace(0,2,21);

% Initialize the table for results
results = zeros(size(alphas));

% Set the number of iterations for K-SVD and for Patch-Disagreement
num_iters_ksvd = 20;
num_iters_disagreement = 30;

% Set weight for noisy image for averaging
mu = 0.005;

%% Part B: Searching for best alpha between [0,2] for all test images

for image_num = 1 : size(images,2)
    % Read an image
    im = imread(images(image_num)); 

    % Convert to double
    im = double(im);

    % Add noise to the input image
    noise = sigma * randn(size(im));
    noisy_im = im + noise;

    % Compute the PSNR of the noisy image and print its value
    psnr_noisy = compute_psnr(im, noisy_im);
    fprintf('PSNR of the noisy image %s is %.3f\n\n',images(image_num), psnr_noisy);
    
    % Run K-SVD
    [im_res_ksvd, D_res_ksvd] = k_svd(noisy_im, D_DCT, patch_size, num_iters_ksvd, sigma, gain_ksvd, mu);

    % Compute the PSNR of the resulting image and print its value
    psnr_ksvd = compute_psnr(im, im_res_ksvd);
    fprintf('PSNR of %s after K-SVD is %.3f\n\n',images(image_num), psnr_ksvd);
    
    % Iterate through various alphas and sum in order to get average after
    for alpha_num = 1:size(alphas,2)
        fprintf('########### Testing for %s and alpha %.3f #############\n',images(image_num), alphas(alpha_num));

        % Resulting dictionary from K-SVD given as initial to Patch-Disgareement,
        % as it done in paper
        [im_res_disagreement, D_res_disagreement] = disagreement(noisy_im, D_res_ksvd, patch_size, num_iters_disagreement, sigma, gain_disagreement, mu, alphas(alpha_num));

        % Compute the PSNR of the resulting image and print its value
        psnr_disagreement = compute_psnr(im, im_res_disagreement);
        fprintf('PSNR of %s after Patch-Disagreement with alpha %.3f is %.3f\n\n\n',images(image_num), alphas(alpha_num), psnr_disagreement);

        % Save results
        results(1,alpha_num) = results(1,alpha_num) + (psnr_disagreement - psnr_ksvd);

        fprintf('For %s and alpha %.3f achieved improvement of %.3f in PSNR!\n',images(image_num), alphas(alpha_num), psnr_disagreement-psnr_ksvd);

    end
end

%% Part C: Show average result

results = results/size(images,2);

figure(1);
plot(alphas,results);
title('Improvement for various alpha values, average');
xlabel('\alpha');
ylabel('PSNR improvement');

