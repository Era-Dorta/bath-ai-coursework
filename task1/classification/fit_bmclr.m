% Description: Multi-class logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        X_test - a (D+1)xI data matrix containing training examples for which
%                 we need to make predictions,
%        num_classes - number of classes.
% Output: Predictions - (num_classes)xI_test matrix which contains the
%                       predicted class values for the data in X_test.
function Predictions = fit_bmclr (X, w, prior, X_test, num_classes)
    % Optimize for phi.
    D1 = size(X,1);
    options = optimset('GradObj','on','Hessian','on');
    initial_phi = ones(D1*num_classes, 1)/num_classes;
    phi = fminunc(@(phi) fit_bmclr_cost(phi, X, w, prior, num_classes),...
        initial_phi, options);

    Phi = reshape(phi,D1,num_classes);
    
    %% Laplace approximation: Evaluate the Hessian at phi hat
    Phi_X_exp = exp(Phi' * X);
    Phi_X_exp_sums = 1 ./ sum(Phi_X_exp,1);
    Y = bsxfun(@times, Phi_X_exp, Phi_X_exp_sums);

    num_test = size(X_test, 2);
    sigma_a = zeros(num_test, num_classes);
    Predictions = zeros(num_classes, num_test);

    mu_a = Phi' * X_test;

    inv_prior = diag(repmat(1/prior,1,D1));
    for n = 1:num_classes
        % Hessian for the current class, taken from the vectorized
        % hessian calculation in fit_mclr_bayesian_cost
        % ddirac(n - n) = 1
        H = X_test * diag(Y(n,:)' .* (1 - Y(n,:)')) * X_test' + inv_prior;
        sigma_a(:,n) = sqrt(diag(X_test' * (H\X_test)));
    end

    %% Monte Carlo integration
    N = 10000; % number of samples
    inv_N = 1 / N;
    for i = 1:num_test
        a_samp = bsxfun(@plus, diag(sigma_a(i,:)) * randn(num_classes,N), mu_a(:,i));
        a_exp = exp(a_samp);
        a_exp_sums = 1./sum(a_exp, 1);
        a_softmax = bsxfun(@times, a_exp, a_exp_sums);
        Predictions(:,i) = inv_N * sum(a_softmax,2);
    end
end
