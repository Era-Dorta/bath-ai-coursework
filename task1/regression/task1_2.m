%% Clear all
clear all;
close all;

%% Load data

load('data/Fonts_n_to_m.mat');

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

colors = ['b','g','r','c','m','y','k', 'b','g','r','c','m','y','k'];

w = Y(trainingIndices,:);

nu = 0.0005;

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 2);

%% Fit model

[mu_test, var_test, relevant] = fit_rvr (X_train, w, nu, X_test, kernel);

%% Check predictions

Y_test = Y(testIndices, :);

% figure;
% for i=1:size(X, 1)
%     plotCharacter(X(i, :), strcat(colors(i),'-'));
% end
% 
% figure;
% for i=1:size(X, 1)
%     plotCharacter(Y(i, :), strcat(colors(i),'-'));
% end