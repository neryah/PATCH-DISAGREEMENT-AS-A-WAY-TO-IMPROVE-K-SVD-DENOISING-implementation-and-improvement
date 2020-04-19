function show_dictionary(D)
% SHOW_DICTIONARY Display a dictionary
%
% Input:
%  D - input dictionary to visualize
%

n_images = 1;
in_D_size = size(D , 1);
each_D_size = in_D_size / n_images;
all_mats = [];

for c1 = 1 : n_images
	D = D((c1 - 1) * each_D_size + (1 : each_D_size) , :);
	
	n_atoms = size(D , 2);

	% Adding borders between the atoms
	atom_size = size(D,1); block_size = round(atom_size .^ 0.5);
	in_inds = (1 : atom_size);
	out_inds = repmat((0 : block_size - 1) * (block_size + 1) , block_size , 1) + repmat((1 : block_size)' , 1 , block_size); out_inds = out_inds(:);
	D2 = zeros((block_size + 1) .^ 2 , n_atoms);
	
	D2(out_inds , :) = D(in_inds , :);
	remInds = setdiff((1 : size(D2 , 1)) , out_inds);
	D2(remInds , :) = -1;
	block_size = block_size + 1;

	Dict = D2;
	
    r = round(n_atoms .^ 0.5); c = r;

	final_mat = zeros([r c] * block_size);
	
	t1 = reshape(Dict(:) , block_size , []); % In this matrix, every blocks adjacent rows (no overlaps) are one block

	inds = (1 : c * block_size);
	for t = 1 : r
		final_mat((t - 1)  * block_size + 1 : t * block_size , :) = t1(: , inds + (t-1) * length(inds));
	end

	s = size(all_mats , 2);

    barrier = [-ones([s 1]) ones([s 1]) -ones([s 2])]';

	if c1 == 1, barrier = []; end
	if isempty(all_mats)
		if ~isempty(barrier)
			all_mats = [barrier ; final_mat];
		else
			all_mats = final_mat;
		end
	else
		all_mats = [all_mats ; barrier ; final_mat];
	end
	
end


rng = max(abs(D(:))); rng = min(rng * 1.1 , 1);

% figure; 
multSize = floor(600 / size(all_mats , 1));
all_mats = imresize(all_mats , multSize , 'nearest');
imshow(all_mats , [-rng rng]);