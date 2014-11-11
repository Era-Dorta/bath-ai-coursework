close all;
clear all;

%% Load data

load('data/MNIST_Data.mat');

num_classes = length(unique(Y));
n_train = 1500;
n_test = 1500;

prior = 100;

% Pick the first n_train samples for training
X_train = X(1:n_train, :);
Y_train = Y(1:n_train);

% And the next n_test samples for testing
X_test = X(n_train + 1:n_train + n_test, :);
Y_test = Y(n_train + 1:n_train + n_test);


% Format the data for fit_mclr_bayesian
data_n_dims = size(X_train, 2);

% When the data has a lot of zeros sparse matrices improve efficiency
X_train_mclr = sparse([ones(1,size(X_train,1)); X_train']);
X_test_mclr = sparse([ones(1,size(X_test,1)); X_test']);
w = Y_train + 1;

%% Profiling
do_profile = 0;
do_time = 1;

if do_profile
    profile clear;
    profile on;
end

if do_time
    tic;
end

%% Get predictions

% Fit a bayesian multi-class logistic regression model
Predictions = fit_mclr_bayesian(X_train_mclr, w, prior, X_test_mclr, num_classes);

if do_time
    toc;
end

if do_profile
    profile off;
    profile viewer;
end

%% Accuracy
[~, predictions_class] = max(Predictions);
predictions_class = (predictions_class - 1)';

compare = Y_test - predictions_class;
accuracy = sum(compare(:) == 0)/n_test;
disp(accuracy);
