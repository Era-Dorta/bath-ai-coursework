%% Clear all
clear all;
close all;

%% Load data

data_path = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat'};
data_index = 1;

load(data_path{data_index});

num_classes_test = length(unique(Y));
num_classes = 2;
precision = 0.01;

if data_index == 1    
    % The class index in this data set starts at zero, increment for
    % matlab indexing
    Y = Y + 1;
end

%% Fit MoG using our function fit_mog.
[lambda, mu, sig] = fit_mog (X(1:10,:), num_classes, precision);

