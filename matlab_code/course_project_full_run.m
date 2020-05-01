%% Description
% This script demonstrates the result of comparison of K-SVD algorithm
% and Algorithm 1 i.e. "Sharing the Disagreement" from paper:
% "PATCH-DISAGREEMENT AS A WAY TO IMPROVE K-SVD DENOISING"
% by Elad and Romano, in a way they did in original paper.
% For every one testing image - 
% "barbara.png","boat.png","couple.bmp","fingerprint.png","house.png","peppers.png"
% it checks on various levels of sigma - 10,  20,   25,   50,   75,   100
% which algorithm gives better denoising result.
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

% Set the seed for the random generator
seed = 33;
 
% Set a fixed random seed to reproduce the results
rng(seed);

% Set the number of iterations for K-SVD and for Patch-Disagreement
num_iters_ksvd = 20;
num_iters_disagreement = 30;

% Set a fixed testing images set as in paper
images = ["barbara.png","boat.png","couple.bmp","fingerprint.png","house.png","peppers.png"];

% Set a fixed standard-deviations set of the Gaussian noise, and a best
% found gains for them accordingly.
sigmas = [10,  20,   25,   50,   75,   100];
gains =  [1.16, 1.16, 1.16, 1.16, 1.14, 1.12];

% Patch dimensions [height, width]
patch_size = [8 8];

% Initialize the dictionary - unitary or not
% D_DCT = build_dct_unitary_dictionary(patch_size);
D_DCT = odctndict(patch_size(1), 256, 2);

% Set weight for noisy image for averaging
mu = 0.005;

% Set default alpha for disagreement - without proposal
alpha = 1;

% Initialize table for results
res_table = zeros(size(sigmas,2),size(images,2));


%% Part B: Run experiments similar to these from original paper

% Iterate on images
for image_num = 1 : size(images,2)
    fprintf('Start testing for image %s\n',images(image_num));
    % Read an image
    im = imread(images(image_num)); 

    % Convert to double
    im = double(im);
    
    % Iterate and compare on several sigmas
    for sigma_num = 1 : size(sigmas,2)
        % Add noise to the input image
        noise = sigmas(sigma_num) * randn(size(im));
        noisy_im = im + noise;

        [psnr_ksvd, psnr_disagreement] = ...
            compare_ksvd_and_disagreement(im, noisy_im, D_DCT, patch_size, sigmas(sigma_num), gains(sigma_num), num_iters_ksvd, num_iters_disagreement, mu, alpha);
        curr_improvement =  psnr_disagreement-psnr_ksvd;
        % Save and print result
        res_table(sigma_num, image_num) = curr_improvement;
        fprintf("For sigma=%i on image %s: K-SVD - %4.2f, PD - %4.2f, achieved improvement of %4.2f!\n",  sigmas(sigma_num), images(image_num), psnr_ksvd, psnr_disagreement, curr_improvement);
    end
end

% Print average improvement
for sigma_num = 1 : size(sigmas,2)
   sigma_avg = sum(res_table(sigma_num,:))/size(images,2); 
   fprintf("\n\nFor sigma=%i achieved average improvement of %4.2f!\n",  sigmas(sigma_num), sigma_avg);
end


