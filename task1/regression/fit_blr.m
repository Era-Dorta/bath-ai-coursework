% Description: Bayesian linear regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions.
% Output: mu_test - a vector of size (I_test x 1), such that mu_test(i)
%                   is the mean of the distribution P(w|x*) for the test
%                   data example x* = X_test(:,i),
%         var_test - a vector of size (I_test x 1), such that var_test(i)
%                    is the variance of the distribution P(w|x*) for the 
%                    test data example x* = X_test(:,i),
%         var - can later be used for plotting the posterior over phi,
%         A_inv - can later be used for plotting the posterior over phi.
function [mu_test, var_test, var, A_inv] = fit_blr (X, w, var_prior, X_test)
    D = size(X,1) - 1;
    I = size(X,2);
    I_test = size(X_test,2);
    
    % Compute the variance. Use the range [0,variance of world values].
    % Constrain var to be positive, by expressing it as var=sqrt(var)^2,
    % that is, the standard deviation squared.
    mu_world = sum(w) / I;
    var_world = sum((w - mu_world) .^ 2) / I;
    var = fminbnd (@(var) fit_blr_cost (var, X, w, var_prior), 0, var_world);
    
    % Compute A_inv.    
    A_inv = 0;    
    if D < I
        A_inv = inv ((X*X') ./ var + eye(D+1) ./ var_prior);
    else    
        A_inv = eye(D+1) - X*inv(X'*X + (var/var_prior)*eye(I))*X';
        A_inv = var_prior * A_inv;
    end

    % Compute the mean for each test example.
    temp = X_test' * A_inv;
    mu_test = (temp * X * w) ./ var;
    
    % Compute the variance for each test example.    
    var_test = repmat(var,I_test,1);
    for i = 1 : I_test
        var_test(i) = var_test(i) + temp(i,:) * X_test(:,i);
    end
end