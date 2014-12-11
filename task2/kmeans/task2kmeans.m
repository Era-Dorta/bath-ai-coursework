%% Clear all
clear all;
close all;

%% Load data

data_path = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat'};
data_index = 1;

load(data_path{data_index});

% Set random seed for repeatable results
rand_seed = 'default';
%rand_seed = 1;
%rand_seed = 10;

rng(rand_seed);

%Uncomment to work with less data
X = X(1:100, :);
Y = Y(1:100);

num_data = size(X, 1);
% Number of classes in our data 
num_classes_test = length(unique(Y));

% Number of gaussians used for the data
if data_index == 1
    num_classes = 10;
else
    num_classes = 8;
end

precision = 0.01;

if data_index == 1
    % The class index in this data set starts at zero, increment for
    % matlab indexing
    Y = Y + 1;
end

%% Fit MoG using our function fit_mog.
[mu, assignm] = kmeans(X, num_classes);

%% Check accuracy

% Rows cluster number, columns votes for class
kmeans_class_vote = zeros(num_classes, num_classes_test);

% Each data will vote that its cluster classifies its class
for i=1:num_data
    kmeans_class_vote(assignm(i), Y(i)) = kmeans_class_vote(assignm(i), Y(i)) + 1;
end

% Each cluster will be associated with the class with max votes
[~, kmeans_real_class] = max(kmeans_class_vote');

% Substitute the cluster number for class number
predictions_class = arrayfun(@(x) kmeans_real_class(x), assignm);

% Check num hits
array_correct_pred = Y - predictions_class';
hits = (sum(array_correct_pred == 0)/num_data) * 100;
if isequal(rand_seed, 'default')
    fprintf('Rand seed default, hits: %2.2f \n', hits);
else
    fprintf('Rand seed %d, hits: %2.2f \n', rand_seed, hits);
end