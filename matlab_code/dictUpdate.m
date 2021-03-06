function [D, X, E] = dictUpdate(atoms_num, D, X, E)
    %% Description
    % Dictionary-Update stage from K-SVD algorithm.
    % X is updated together with D.
    % Used as part of K-SVD and "Patch-Disagreement" implementations.
    % 
    % Done as part of course project for course:
    % 236862 Sparse and Redundant Representations
    % Technion - IIT, 2020
    
    %% Parameters:
    % atoms_num - Number of atoms in the dictionary.
    % D - The current dictionary.
    % X - Holds sparse representations.
    % E - Total error matrix .

    % Define w_k as a group of examples which used atom d_k in
    % their sparse representation.
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
end