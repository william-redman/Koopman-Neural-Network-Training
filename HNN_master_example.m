%% HNN Master Example
%-------------------------------------------------------------------------%
%   This script runs all the analysis for the Hamiltonian NN feedforward 
%   network sample data. The code for the HNN whose weights and biases 
%   these came from a modified version of the code present at
%   https://github.com/mariosmat/hamiltonian_networks.
%
%   This script constructs the Node Koopman operators for each of the 25
%   NNs trained using Adadelta. To build these operators, the weight and 
%   bias data from time t1 to t2 is used. The network is then pushed 
%   forward T steps. Note that t1 and t2 are both plus 35000, the time
%   when we starting saving the data. Therefore, t1 = 1 is not early
%   in-training per-se. 
%
%   Once the Node Koopman operators have been constructed and the weights
%   have been evolved T time steps, the predicted weights and biases are
%   compared to the true weights and biases. 
%
%   DEPENDENCIES: Adadelta example data and NodeKoopmanTraining function,
%   both of which can be found at 
%   https://github.com/william-redman/Koopman-Neural-Network-Training.
%
%   Written by WTR 09/21/2020 // Last updatd by WTR 02/13/2021
%-------------------------------------------------------------------------%
%% Globals
t1 = 1;                                                    
t2 = 10000; 
T = 2500;  

if (T + t2) > 15000
    error('Not enough saved weight/bias data to compare prediciton to');
end

n_n = 25;                                               % number of networks
n_h = 10;                                               % number of units per hidden layer
n_p = [ n_h + n_h, n_h ^ 2 + n_h, 2 * n_h + 2 ];        % number of weights and biases per layer
save_flag = 0;                                          % whether you want to save the predicted weights/biases

n_top = 11;                                             % number of best performing networks for plotting

data_path = uigetdir('Select folder with example data');

%% Computing the Koopman trained weights
KO_rt_vec = zeros(1, n_n); 
e1 = zeros(n_n, n_p(1)); 
e2 = zeros(n_n, n_p(2)); 
eout = zeros(n_n, n_p(3)); 
d1 = zeros(n_n, n_p(1)); 
d2 = zeros(n_n, n_p(2));
dout = zeros(n_n, n_p(3)); 

for ii = 1:n_n
    ii
    % Load data
    cd(strcat(data_path, '/Network_', num2str(ii)));  
        
    tic
    load('Weights_Layer_1.mat');
    load('Weights_Layer_2.mat'); 
    load('Weights_Out.mat'); 
    load('Biases_Layer_1.mat'); 
    load('Biases_Layer_2.mat'); 
    load('Biases_Out.mat');
    
    % Computing predictions
    KO_P1 = NodeKoopmanTraining(cat(2, W1(:, :, t1:t2), reshape(b1(:, t1:t2), size(b1, 1), 1, t2 - t1 + 1)), 2, T, 0);
    KO_W1 = KO_P1(:, 1:(end - 1))'; KO_b1 = KO_P1(:, end)'; 
    KO_P2 = NodeKoopmanTraining(cat(2, W2(:, :, t1:t2), reshape(b2(:, t1:t2), size(b2, 1), 1, t2 - t1 + 1)), n_h + 1, T, 0);
    KO_W2 = KO_P2(:, 1:(end - 1)); KO_b2 = KO_P2(:, end)'; 
    KO_Pout = NodeKoopmanTraining(cat(2, Wout(:, :, t1:t2), reshape(bout(:, t1:t2), size(bout, 1), 1, t2 - t1 + 1)), n_h + 1, T, 0);
    KO_Wout = KO_Pout(:, 1:(end - 1)); KO_bout = KO_Pout(:, end)'; 

    % Saving
    if save_flag
        save('KO_W1.mat', 'KO_W1'); 
        save('KO_W2.mat', 'KO_W2');
        save('KO_Wout.mat', 'KO_Wout');
        save('KO_b1.mat', 'KO_b1');
        save('KO_b2.mat', 'KO_b2');
        save('KO_bout.mat', 'KO_bout');
    end    
        
    [KO_training_time] = toc;
    KO_rt_vec(ii) = KO_training_time; 
    
    if save_flag
        save('KO_training_time.mat', 'KO_training_time')
    end
    
    P1 = cat(2, W1, reshape(b1, size(b1, 1), 1, size(b1, 2))); 
    P2 = cat(2, W2, reshape(b2, size(b2, 1), 1, size(b2, 2))); 
    Pout = cat(2, Wout, reshape(bout, size(bout, 1), 1, size(bout, 2))); 
    
    % Computing the error
    dev_1 = KO_P1 - P1(:, :, t2 + T);
    dev_2 = KO_P2 - P2(:, :, t2 + T);
    dev_out = KO_Pout - Pout(:, :, t2 + T);
      
    e1(ii, :) = dev_1(:); 
    e2(ii, :) = dev_2(:); 
    eout(ii, :) = dev_out(:); 
    
    % Computing the distance travelled 
    dist_1 = P1(:, :, t2 + T) - P1(:, :, t2); 
    dist_2 = P2(:, :, t2 + T) - P2(:, :, t2);
    dist_out = Pout(:, :, t2 + T) - Pout(:, :, t2);
    
    d1(ii, :) = dist_1(:); 
    d2(ii, :) = dist_2(:);
    dout(ii, :) = dist_out(:);
    
end

%% Plotting error 
%   Note that this plot is comparable to Fig. 1e, S2e, and S3e in Dogra and
%   Redman NeurIPS 2020. 
e = mean([mean(abs(e1), 2), mean(abs(e2), 2), mean(abs(eout), 2)], 2); 
[~, ids] = sort(e); 
top_ids = ids(1:n_top); 
true_growth = [];
Koop_error = []; 

figure 
for ii = 1:n_top 
    color = [(ii - 1) / (n_top - 1), 0, (n_top - ii) / (n_top - 1)]; 
    
    e_ii = [e1(top_ids(ii), :), e2(top_ids(ii), :), eout(top_ids(ii), :)]; 
    d_ii = [d1(top_ids(ii), :), d2(top_ids(ii), :), dout(top_ids(ii), :)];
    
    true_growth = [true_growth, d_ii]; 
    Koop_error = [Koop_error, e_ii]; 
    
    plot(d_ii, e_ii, 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', color); hold on
end
    
xlabel('True weight evolution');
ylabel('Error');

median_Koop_error = median(abs(Koop_error));
median_true_growth = median(abs(true_growth));
error2growth = median_Koop_error / median_true_growth


