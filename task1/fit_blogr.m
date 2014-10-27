% Author: Stefan Stavrev 2013

% Description: Bayesian logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        initial_phi - (D+1)x1 vector that represents the start solution.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
%         phi - (D+1)x1 column vector that contains the coefficients for
%               the linear activation function.
function [predictions, phi] = fit_blogr (X, w, var_prior, X_test, initial_phi)
    I = size(X,2);
    I_test = size(X_test,2);
    D = size(X,1) - 1;

    % Find the MAP estimate of the parameters phi.
    options = optimset('GradObj','on','Hessian','on');
    
    % X = fminunc(FUN,X0,OPTIONS)
    % Create an anonymous function that takes one argument(phi) and
    % executes the single line fit_logr_cost(phi, X, w, var_prior),
    % so it would actually be executing fit_logr_cost.
    phi = fminunc(@(phi) fit_logr_cost(phi, X, w, var_prior), ...
                  initial_phi, options);
              
% a = [1,2,3];
% repmat(a, 2, 3)
% ans =
% 1     2     3     1     2     3     1     2     3
% 1     2     3     1     2     3     1     2     3

    % Compute the Hessian at phi.
    H = diag(repmat(1/var_prior,1,D+1));
    ys = sigmoid(phi' * X); 
    for i=1:I
        x_i = X(:,i);
        y = ys(i);
        H = H - y * (1-y) * (x_i * x_i');
    end
    
    % Set the mean and variance of the Laplace approximation.
    mu = phi;
    var = -inv(H);
    
    % Compute mean and variance of the activation.
    mu_a = mu' * X_test;
    var_a_temp = X_test' * var;
    var_a = zeros(1,I_test);
    for i=1:I_test
        var_a(i) = var_a_temp(i,:) * X_test(:,i);
    end
    
    % Approximate the integral to get the Bernoulli parameter.
    lambda = sqrt(1 + pi * var_a / 8);    
    lambda = mu_a ./ lambda;
    lambda = sigmoid(lambda);
    
    predictions = lambda;
end