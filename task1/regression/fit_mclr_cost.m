% Author: Stefan Stavrev 2013

% Description: Cost function for multi-class logistic regression.
% Input: phi - a (D+1)*num_classesx1 vector that contains the parameters
%              that are subject to optimization,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        num_classes - number of classes.
% Output: f - the value of the function,
%         g - the gradient,
%         H - the Hessian.
function [L, g, H] = fit_mclr_cost (phi, X, w, num_classes)
    % Init.
    L = 0;
    D1 = size(X,1); 
    D = D1 - 1;
    Phi = reshape(phi,D1,num_classes);
    num_variables = D1 * num_classes;
    g = zeros(num_variables,1);
    H = [];
    I = size(X,2);
    ddirac = @(x) double(not(x));
    HH = cell(num_classes, num_classes);
    for i = 1 : num_classes
        for j = 1 : num_classes
            HH{i,j} = zeros(D1,D1);
        end
    end
    
    % Compute the predictions Y for X.
    Phi_X = Phi' * X;
    Phi_X_exp = exp(Phi_X);
    Phi_X_exp_sums = 1 ./ sum(Phi_X_exp,1);
    Y = bsxfun(@times, Phi_X_exp, Phi_X_exp_sums);
    
    for i = 1 : I
        % Update log likelihood L.
        L = L - log(Y(w(i),i));
        
        start = 1;
        for n = 1 : num_classes
            % Update gradient.
            temp1 = (Y(n,i) - ddirac(w(i)-n)) * X(:,i);
            g(start : start+D) = g(start : start+D) + temp1;
            start = start + D1;
            
            % Update Hessian.            
            for m = 1 : num_classes
                temp2 = Y(m,i) * (ddirac(m-n) - Y(n,i)) * X(:,i) * X(:,i)'; %Slowest lines
                HH{m,n} = HH{m,n} + temp2; %Slow line too
            end
        end
    end
    
    % Assemble final Hessian.
    for n = 1 : num_classes
        H_n = [];
        for m = 1 : num_classes
            H_n = [H_n HH{n,m}];
        end
        H = [H; H_n];
    end
end