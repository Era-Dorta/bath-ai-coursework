%% Clear all
clear all;
close all;

%% Load data

load('../../data/Fonts_n_to_m.mat');

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

colors = ['b','g','r','c','m','y','k', 'b','g','r','c','m','y','k'];

w = Y(trainingIndices,:);

% 3 vectors
%nu = 0.0005;

% 9 vectors
nu = 0.005;

% 5 vectors
%nu = 0.001;

lambda = 2;

plotDev = 0;
plotNormalized = 0;

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, lambda);

%% Fit model

do_profile = 0;

if do_profile
    profile on;
end

[mu_test, var_test, relevant] = fit_rvr (X_train, w, nu, X_test, kernel);

var_test = var_test';

num_relevant = sum(relevant);

fprintf('Using %d relevance vectors\n', num_relevant);

if do_profile
    profile off;
    profile viewer;
end

%% Check predictions

Y_test = Y(testIndices, :);
for i=1:n_test
    figure;
    plotCharacter(Y(testIndices(i), :), 'b-');
    if plotNormalized
        mu_test(i,:) = norm(Y_test(i,:))* mu_test(i,:) / norm(mu_test(i,:));
        var_test(i,:) = norm(Y_test(i,:))* var_test(i,:) / norm(mu_test(i,:));
    end
    plotCharacter(mu_test(i, :), 'r-');
    if plotDev
        standardDeviation = var_test(i,:).^0.5;
        varMax = max(mu_test(i, :) + 2*standardDeviation,mu_test(i, :) - 2*standardDeviation);
        varMin = min(mu_test(i, :) + 2*standardDeviation,mu_test(i, :) - 2*standardDeviation);
        plotCharacter(varMax, 'k.');
        plotCharacter(varMin, 'k.');
    end
    title(strcat('Nu = ',num2str(nu)));
end


% figure;
% for i=1:size(X, 1)
%     plotCharacter(X(i, :), strcat(colors(i),'-'));
% end
%
% figure;
% for i=1:size(X, 1)
%     plotCharacter(Y(i, :), strcat(colors(i),'-'));
% end