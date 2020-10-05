function [ P ] = NodeKoopmanTraining( W, q, T, iFlag )
%% Node Koopman Training
%-------------------------------------------------------------------------%
%   This function performs node Koopman training on D, which can be
%   composed of neural network weights and/or biases. This function was
%   used in Dogra and Redman 2020 arXiv:2006.02361.
%
%   The assumptions here are that be that the input D is a tensor with the
%   weights/biases stored in the first and second dimensions, and the time
%   evolution stored in the third. Additionally, it assumes that the 
%   dimension being "noded" over is the second one (as is convention). 
%
%   The input q determines whether the weights/biases going to each node
%   should be further split up into finer chunks. Setting q = 1 sets them 
%   all into their own chunks and setting q = size(D, 2) sets them all into 
%   a single chunk. 
%
%   Written by WTR 09/21/2020 // Last updated by WTR 09/21/2020
%-------------------------------------------------------------------------%
%% Globals 
n1 = size(W, 1); 
n2 = size(W, 2); 
g = n2 / q;
rem_g = rem(n2, q); 

%% Warnings
if rem_g > 0
    warning('Matrix dimension not divisible by q-factor. Will have one set leftover'); 
end

%% Koopman prediction
if iFlag
    P = zeros(n1, n2, T); 
else
    P = zeros(n1, n2); 
end

for ii = 1:n1
    for jj = 1:floor(g)
        D = squeeze(W(ii, ((jj - 1) * q + 1):(jj * q), :));
        
        if q == 1
            D = D';
        end
        
        F = D(:, 1:(end - 1)); 
        Fp = D(:, 2:end); 

        U = Fp * pinv(F); 
        
        if iFlag 
            P(ii, ((jj - 1) * q + 1):(jj * q), 1) = U * Fp(:, end); 
            for tt = 2:T
                P(ii, ((jj - 1) * q + 1):(jj * q), tt) = U * squeeze(P(ii, ((jj - 1) * q + 1):(jj * q), tt - 1))'; 
            end
        else                               
            P(ii, ((jj - 1) * q + 1):(jj * q)) = U ^ T * Fp(:, end);
        end
    end
    
    if rem_g > 0
        D = squeeze(W(ii, (end - rem_g + 1):end, :)); 
        
        if rem_g == 1
            D = D';
        end
        
        F = D(:, 1:(end - 1)); 
        Fp = D(:, 2:end); 

        U = Fp * pinv(F); 
        
        if iFlag 
            P(ii, (end - rem_g + 1):end, 1) = U * Fp(:, end); 
            for tt = 2:T
                P(ii, (end - rem_g + 1):end, tt) = U * squeeze(P(ii, (end - rem_g + 1):end, tt - 1))'; 
            end
        else                                     
            P(ii, (end - rem_g + 1):end) = U ^ T * Fp(:, end);
        end
        
    end
end


end
