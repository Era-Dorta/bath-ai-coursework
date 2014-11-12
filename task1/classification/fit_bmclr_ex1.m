close all;
clear all;

%% Load data

% {Digits, eth-80, sport_bow} = {1, 2, 3}
data_index = 1;

data_paths = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat', 'data/UIUCSport_BoW_Data.mat'};

load(data_paths{data_index});

num_classes = length(unique(Y));
n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

if data_index == 1
    prior = 1;
    % Pick the first n_train samples for training
    X_train = X(1:n_train, :);
    Y_train = Y(1:n_train);

    % And the next n_test samples for testing
    X_test = X(n_train + 1:n_train + n_test, :);
    Y_test = Y(n_train + 1:n_train + n_test);
    
    % The class index in this data set starts at zero, increment for
    % matlab indexing
    w = Y_train + 1;
end

if data_index == 2 || data_index == 3
    prior = 1000;
    % Pick trainingIndices samples for training
    X_train = X(trainingIndices, :);
    Y_train = Y(trainingIndices);

    % Pick testIndices samples for testing
    X_test = X(testIndices, :);
    Y_test = Y(testIndices);
    w = Y_train;
end

% Format the data for fit_bmclr
X_train_bmclr = [ones(1,size(X_train,1)); X_train'];
X_test_bmclr = [ones(1,size(X_test,1)); X_test'];

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
Predictions = fit_bmclr(X_train_bmclr, w, prior, X_test_bmclr, num_classes);

if do_time
    total_time = toc
end

if do_profile
    profile off;
    profile viewer;
end

%% Accuracy measurement
% Predicted class is the one with the highest probability score
[~, predictions_class] = max(Predictions);

if data_index == 1
    % Decrease class index number back to original
    predictions_class = (predictions_class - 1);
end

% Get percentage of correct predictions
array_correct_pred = Y_test - predictions_class';
hits = (sum(array_correct_pred == 0)/n_test) * 100;
fprintf('Hits: %2.2f%%\n', hits);
