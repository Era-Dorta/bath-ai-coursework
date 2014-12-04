% Author: Stefan Stavrev 2013

% Description: Relevance vector regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        nu - degrees of freedom typically nu<0.001,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        kernel - the kernel function.
% Output: mu_test - a vector of size (I_test x 1), such that mu_test(i)
%                   is the mean of the distribution P(w|x*) for the test
%                   data example x* = X_test(:,i),
%         var_test - a vector of size (I_test x 1), such that var_test(i)
%                    is the variance of the distribution P(w|x*) for the
%                    test data example x* = X_test(:,i),
%         relevant_points - Ix1 boolean vector where a 1 at position i
%                           indicates that point X(:,i) remained after
%                           the elimination phase, that is, it is relevant.
function [mu_test, var_test, relevant_points] = ...
    fit_rvr_sameH (X, w, nu, X_test, kernel)
I = size(X,2);
I_test = size(X_test,2);
D = size(X, 1) - 1;

% Compute K[X,X].
K = zeros(I,I);
for i=1:I
    for j=1:I
        K(i,j) = kernel(X(:,i), X(:,j));
    end
end

% Initialize H.
H = ones(I,1);
H_old = zeros(I,1);

% Pre-compute so that it is not computed each iteration.
K_K = K*K;
K_w = K*w;

% The main loop.
iterations_count = 0;
precision = 1e-5;
min_H = ones(I,1);
min_Cost = 0;
minCount = 0;
maxCount = 15;
firstLoop = 1;
maxPrecision = 200;

var = zeros(D, 1);
while true
    % Compute the variance. Use the range [0,variance of world values].
    % Constrain var to be positive, by expressing it as
    % var=sqrt(var)^2, that is, the standard deviation squared.
    %8.56
    for i=1:D
        mu_world = sum(w(:,i)) / I;
        var_world = sum((w(:,i) - mu_world) .^ 2) / I;
        var(i) = fminbnd (@(var) fit_rvr_cost (var, K, w(:,i), H(:,1)), 0, var_world);
        
        %Equation 8.57
        % Update sig and mu.
        sig = K_K/var(i) + diag(H(:,1));
        %sig = inv (K_K/var(i) + diag(H(:,i)));
        mu = sig\(K_w(:,i)/var(i));
        sig = inv(sig);
        
    end
    
    hFunc = @(H_In) evalCostSameH(var,K,w,H_In,D);
    H = fminsearch(hFunc,H,optimset('Display','final','MaxIter',50));
    
    iterations_count = iterations_count + 1;
    disp(['iteration ' num2str(iterations_count)]);
    
    testCost = evalCostSameH(var, K, w, H, D);
    disp(['Current Cost: ' num2str(testCost)]);
    if ~firstLoop && (min_Cost == 0 || testCost < min_Cost)
        min_Cost = testCost;
        min_H = H;
        minCount = 0;
    else
        minCount = minCount + 1;
    end
    
    disp(['Current minCount: ' num2str(minCount)]);
    
    if minCount > maxCount
        disp('Solution looping, taking local minima');
        H = min_H;
        break;
    end
    
    current_precision = mean(mean(abs(H-H_old)));
    disp(['Current Precision: ' num2str(current_precision)]);
    
    stop = (current_precision < precision || current_precision > maxPrecision);
    %stop = all(abs(H-H_old) < precision);
    if stop
        break;
    end
    
    % Save H for the next iteration.
    
    H_old = H;
    
    firstLoop = 0;
end

fprintf('H total values %d, H smaller than precision %d\n', ...
    size(H,1)*size(H,2), sum(sum(abs(H-H_old) < precision)));
assignin('base','H',H);

% Prune step. Remove column t in X, row t in w, and element t in H,
% if H(t) > Some Threshold.

selector = H < 1000;
X = X(:,selector);
w = w(selector, :);
H = H(selector, :);
relevant_points = selector;

% Recompute K[X,X].
I = size(X,2);
K = zeros(I,I);
for i=1:I
    for j=1:I
        K(i,j) = kernel(X(:,i), X(:,j));
    end
end

% Compute K[X_test,X].
K_test = zeros(I_test, I);
for i=1:I_test
    for j=1:I
        K_test(i,j) = kernel(X_test(:,i), X(:,j));
    end
end

for i=1:D
    % Compute A_inv.
    A_inv = (K*K/var(i) + diag(H(:,1)));
    
    % Compute the mean for each test example.
    % Use "b/A rather than b * inv(A)"
    temp = K_test/A_inv;
    mu_test(:,i) = temp*K*w(:,i)/var(i);
    
    % Compute the variance for each test example.
    var_test2 = repmat(var(i),I_test,1);
    for j = 1 : I_test
        var_test(i,j) = var_test2(j) + temp(j,:)*K_test(j,:)';
    end
end
end