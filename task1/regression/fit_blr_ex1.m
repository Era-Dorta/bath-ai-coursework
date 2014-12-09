%% Load data
data_path = 'data/Fonts_n_to_m.mat';

load(data_path);

var_prior = 6;

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

w = Y(trainingIndices,:);

D = size(X,2);

mu_test = zeros(n_test, D);
% Fit Bayesian linear regression model.
for d=1:D
    [mu_test(:,d), var_test, var, A_inv] = fit_blr (X_train(d,:), w(:,d), var_prior, X_test(d,:));
end


plotNormalized = 0;
plotDev = 0;
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
end

