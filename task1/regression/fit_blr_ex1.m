clear all;
close all;

%% Load data
data_path = 'data/Fonts_n_to_m.mat';

load(data_path);

var_prior = 0.1;

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

w = Y(trainingIndices,:);

D = size(X,2);

mu_test = zeros(n_test, D);
var_test = zeros(n_test, D);

% Fit Bayesian linear regression model for each dimension in w.
for d=1:D
    [mu_test(:,d), var_test(:,d), ~, ~] = fit_blr (X_train, w(:,d), var_prior, X_test);
end

plotDev = 0;

Y_test = Y(testIndices, :);
for i=1:n_test
    figure;
    plotCharacter(Y(testIndices(i), :), 'b-');
    plotCharacter(mu_test(i, :), 'r-');
    
    if plotDev
        standardDeviation = 0.1 * var_test(i,:).^0.5;
        varMax = mu_test(i, :) + standardDeviation;
        varMin = mu_test(i, :) - standardDeviation;
        plotCharacter(varMax, 'k.');
        plotCharacter(varMin, 'k.');
    end
end

