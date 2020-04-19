clear; clc; close all;

% Set the seed for the random generator
seed = 33;
 
% Set a fixed random seed to reproduce the results
rng(seed);

% Set K - the cardinality of the solution.
% This will serve us later as the threshold for the OMP
K = 4;

% Set the number of iterations for K-SVD and for Patch-Disagreement
num_iters_ksvd = 20;
num_iters_disagreement = 30;

% Set a fixed testing images set as in paper
images = ["barbara.png","boat.png","couple.bmp","fingerprint.png","house.png","peppers.png"];

% Set a fixed standard-deviations set of the Gaussian noise
sigmas = [10, 20, 25, 75, 100];

% Patch dimensions [height, width]
patch_size = [8 8];

% Initialize the dictionary - unitary or not
% D_DCT = build_dct_unitary_dictionary(patch_size);
D_DCT = odctndict(patch_size(1), 256, 2);

% Iterate on images
for image_num = 1 : 6
    fprintf('Start testing for image %s\n',images(image_num));
    % Read an image
    im = imread(images(image_num)); 

    % Convert to double
    im = double(im);
    
    % Iterate and compare on several sigmas
    for sigma_num = 1 : 5
        % Add noise to the input image
        noise = sigmas(sigma_num) * randn(size(im));
        noisy_im = im + noise;

        [psnr_ksvd, psnr_disagreement] = ...
            compare_ksvd_and_disagreement(im, noisy_im, D_DCT, patch_size, K, num_iters_ksvd, num_iters_disagreement);
        fprintf("For sigma=%i on image %s: K-SVD - %4.4f, PD - %4.4f, achieved improvement of %4.4f!\n",  sigmas(sigma_num), images(image_num), psnr_ksvd, psnr_disagreement, psnr_disagreement-psnr_ksvd);
    end 
    
end

