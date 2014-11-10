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
function Predictions = fit_mclr_bayesian (X, w, prior, X_test, num_classes)
    % Optimize for phi.
    D1 = size(X,1);
    options = optimset('GradObj','on','Hessian','on');
    initial_phi = ones(D1*num_classes, 1);
    phi = fminunc(@(phi) fit_mclr_bayesian_cost(phi, X, w, prior, num_classes),...
        initial_phi, options);

	Phi = reshape(phi,D1,num_classes);

	%% Compute the Hessian at phi_hat and Predict
	Phi_X_exp = exp(Phi' * X);
	Phi_X_exp_sums = 1 ./ sum(Phi_X_exp,1);
	Y = bsxfun(@times, Phi_X_exp, Phi_X_exp_sums);


	%%
	I = size(X_test,2);
	var_a = zeros(I,num_classes);
	sigma_a = zeros(I,num_classes);
	Predictions = zeros(num_classes,I);

	mu_a = Phi' * X;

    inv_prior = diag(repmat(1/prior,1,D1));
	for n = 1:num_classes
	    % Get Hessian for the one class
        % ddirac(n - n) = 1
	    H = X * diag(Y(n,:)' .* (1 - Y(n,:)')) * X' + inv_prior;
	    var_a(:,n) = diag(X' * (H\X));
	    sigma_a(:,n) = sqrt(var_a(:,n));
	end

	N = 10000; % number of samples
	for i = 1:I
	    % Monte Carlo integration
	    a_samp = bsxfun(@plus, diag(sigma_a(i,:))*randn(num_classes,N), mu_a(:,i));
	    a_exp = exp(a_samp);
	    a_exp_sums = 1./sum(a_exp,1);
	    a_softmax = bsxfun(@times, a_exp, a_exp_sums);
	    Predictions(:,i) = (1/N) * sum(a_softmax,2);
	end
end
