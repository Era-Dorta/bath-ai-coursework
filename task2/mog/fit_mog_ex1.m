%% Clear all
clear all;
close all;

%% Load data

data_path = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat'};
data_index = 1;

load(data_path{data_index});

%Uncomment to work with less data
%X = X(1:100, :);
%Y = Y(1:100);

num_data = size(X, 1);
num_classes_test = length(unique(Y));
num_classes = 10;
precision = 0.01;

if data_index == 1    
    % The class index in this data set starts at zero, increment for
    % matlab indexing
    Y = Y + 1;
end

%% Fit MoG using our function fit_mog.
[lambda, mu, sig, r] = fit_mog (X, num_classes, precision);


%% Check accuracy
[~, predictions_class] = max(r');

% Rows gaussian number, columns votes for class
gaussians_class_vote = zeros(num_classes, num_classes_test);

% Each data will vote that its Gaussian classifies its class
for i=1:num_data
    gaussians_class_vote(predictions_class(i), Y(i)) = gaussians_class_vote(predictions_class(i), Y(i)) + 1;
end

% Transform votes to probabilities
gaussians_class_vote_norm = bsxfun(@rdivide, gaussians_class_vote, sum(gaussians_class_vote)); 

% Gaussian will classify the class with max votes
%[~, gaussian_real_class] = max(gaussians_class_vote_norm');

class_index = 1:num_classes_test;
gaussian_index = 1:num_classes;
gaussian_real_class = zeros(1, num_classes);
gaussians_class_vote_norm1 = gaussians_class_vote_norm;
for i=1:num_classes
    % Get the gaussian with the maximum probability for these class
    [~, gaussian_real_class(i)] = max(gaussians_class_vote_norm(gaussian_index, i));
    
    % Set the probability to zero so these Gaussian does not get picked
    % again
    gaussians_class_vote_norm(gaussian_real_class(i),:) = 0;
end

% Subsititute the gaussian number for class number
predictions_class = arrayfun(@(x) gaussian_real_class(x), predictions_class);

% Check num hits
array_correct_pred = Y - predictions_class';
hits = (sum(array_correct_pred == 0)/num_data) * 100;
fprintf('Data %s, hits: %2.2f%%\n', data_path{data_index}, hits);