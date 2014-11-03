% Author: Stefan Stavrev 2013

% Description: Multi-class logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        num_classes - number of classes.
% Output: Predictions - (num_classes)xI_test matrix which contains the
%                       predicted class values for the data in X_test.
function Predictions = fit_mclr (X, w, X_test, num_classes)
    % Optimize for phi.
    D1 = size(X,1);
    options = optimset('GradObj','on','Hessian','on');
    initial_phi = ones(D1*num_classes, 1);
    phi = fminunc(@(phi) fit_mclr_cost(phi, X, w, num_classes),...
        initial_phi, options);
    Phi = reshape(phi,D1,num_classes);
    
    % Predict.
    Phi_X_exp = exp(Phi' * X_test);
    Phi_X_exp_sums = 1 ./ sum(Phi_X_exp,1);
    Predictions = bsxfun(@times, Phi_X_exp, Phi_X_exp_sums);
end