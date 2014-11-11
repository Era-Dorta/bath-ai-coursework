close all;
clear all;

%% Load data

load('data/MNIST_Data.mat');

num_classes = length(unique(Y));
n_train = 50;
n_test = 50;

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
% Increase class index number for matlab easy indexing
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
Predictions = fit_bmclr(X_train_mclr, w, prior, X_test_mclr, num_classes);

if do_time
    toc;
end

if do_profile
    profile off;
    profile viewer;
end

%% Accuracy
% Prediction is the one with the highest probability
[~, predictions_class] = max(Predictions);

% Decrease class index number back
predictions_class = (predictions_class - 1)';

% Get percentage of correct predictions
array_correct_pred = Y_test - predictions_class;
hits = (sum(array_correct_pred == 0)/n_test) * 100;
fprintf('Hits: %2.2f%%\n', hits);
