clear all;
close all;

%% Load data
data_path = 'data/Fonts_n_to_m.mat';

load(data_path);

var_prior = [10, 1, 0.1, 0.01];

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

w = Y(trainingIndices,:);

D = size(X,2);

mu_test = zeros(n_test, D);
var_test = zeros(n_test, D);

for i=1:length(var_prior)
    % Fit Bayesian linear regression model for each dimension in w.
    for d=1:D
        [mu_test(:,d), var_test(:,d), ~, ~] = fit_blr (X_train, w(:,d), var_prior(i), X_test);
    end
    
    plotDev = 0;
    close all;
    
    Y_test = Y(testIndices, :);
    for j=1:n_test
        figure;
        plotCharacter(Y(testIndices(j), :), 'b-');
        plotCharacter(mu_test(j, :), 'r-');
        
        if plotDev
            standardDeviation = 0.1 * var_test(j,:).^0.5;
            varMax = mu_test(j, :) + standardDeviation;
            varMin = mu_test(j, :) - standardDeviation;
            plotCharacter(varMax, 'k.');
            plotCharacter(varMin, 'k.');
        end
    end
    
    fprintf('Results usign prior variance %2.2f\n', var_prior(i));
    
    prompt = 'Continue with next variance? Y/N [Y]: ';
    cont = input(prompt,'s');
    if isempty(cont)
        cont = 'Y';
    end
    
    if ~isequal(cont, 'Y')
        break;
    end
end
