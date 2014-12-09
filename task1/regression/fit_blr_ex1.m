%% Load data
data_path = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat'};
data_index = 1;

load(data_path{data_index});

num_classes = length(unique(Y));
n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

var_prior = 6;

if data_index == 1
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
    % Pick trainingIndices samples for training
    X_train = X(trainingIndices, :);
    Y_train = Y(trainingIndices);

    % Pick testIndices samples for testing
    X_test = X(testIndices, :);
    Y_test = Y(testIndices);
    w = Y_train;
end

% Format the data for fit_bmclr
X_train_blr = [ones(1,size(X_train,1)); X_train'];
X_test_blr = [ones(1,size(X_test,1)); X_test'];

% Fit Bayesian linear regression model.
[mu_test, var_test, var, A_inv] = fit_blr (X_train_blr, w, var_prior, X_test_blr);
