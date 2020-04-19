function im = col_to_im(patches, patch_size, im_size)
% COL_TO_IM Rearrange matrix columns into an image of size MXN
%
% Inputs:
%  patches - A matrix of size 
%            (patch_size(1)*patch_size(2)) X (M-patch_size(1)+1)*(N-patch_size(2)+1)
%  patch_size - The size of the patch [height width]
%  im_size    - The size of the image we aim to build [height width] = [M N]
%
% Output:
%  im - The reconstructed image, computed by returning the patches in
%       'patches' to their original locations, followed by a
%       patch-averaging over the overlaps
%

inds = reshape(1 : prod(im_size), im_size);
inds = im2col(inds,[patch_size patch_size]);
im = accumarray(inds(:), patches(:), [prod(im_size) 1], @(x) mean(x));
im = reshape(im, im_size);

end