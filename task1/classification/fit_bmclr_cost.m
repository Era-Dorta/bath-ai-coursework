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
function [L, g, H] = fit_bmclr_cost (phi, X, w, prior, num_classes)
    % Initialization
    %% Adding prior to log likelihood
    L = -1 / (2 * prior) * (phi' * phi);
    D1 = size(X,1);
    D = D1 - 1;
    Phi = reshape(phi,D1,num_classes);
    num_variables = D1 * num_classes;
    g = zeros(num_variables,1);
    H = [];
    I = size(X,2);
    ddirac = @(x) double(not(x));

    % Create a num_class by num_class cell of hessian matrices
    HH = cell(num_classes, num_classes);
    % Create a num_class by num_class cell of indices for the hessian
    index_mat_cell = cell(num_classes, num_classes);

    for i = 1 : num_classes
        for j = 1 : num_classes
            HH{i,j} = zeros(D1,D1);
            index_mat_cell{i, j} = [i, j];
        end
    end

    %% Compute the predictions Y for X.
    Phi_X = Phi' * X;
    Phi_X_exp = exp(Phi_X);
    Phi_X_exp_sums = 1 ./ sum(Phi_X_exp,1);
    Y = bsxfun(@times, Phi_X_exp, Phi_X_exp_sums);

    %% Log likelihood and gradient update loop
    % It is not vectorized because because the hessian update is the slowest
    % part by far
    for i = 1 : I
        % Update log likelihood L.
        L = L - log(Y(w(i),i));

        start = 1;
        for n = 1 : num_classes
            % Update gradient.
            temp1 = (Y(n,i) - ddirac(w(i)-n)) * X(:,i);
            g(start : start+D) = g(start : start+D) + temp1;
            start = start + D1;

            % Update Hessian, left for reference of vectorization process
            % for m = 1 : num_classes
            %  temp2 = Y(m,i) * (ddirac(m-n) - Y(n,i)) * XbyXtras; 
            %  HH{m,n} = HH{m,n} + temp2;
            % end

            %Vectorized v1 version of Hessian update
            % class_index = num2cell(1:num_classes)';
            %
            % HH(:, n) = cellfun(@(x, m) x + Y(m,i) * (ddirac(m-n) - Y(n,i)) ...
            %     * XbyXtras, HH(:, n), class_index, 'UniformOutput', false);
        end

        %Vectorized v2 version of Hessian update
        % HH = cellfun(@(x, ind)  x + X(:, i) * Y(ind(1),i) * (ddirac(ind(1)-ind(2)) ...
        %  - Y(ind(2),i)) * X(:, i)', ...
        %  HH, index_mat_cell, 'UniformOutput', false);
    end
    
    % Extra gradient term for bayesian classification
    phi_over_prior = Phi/prior;
    g = g + phi_over_prior(:);   
    
    %% Update hessian
    
    %Vectorized v3 version of Hessian update
    inv_prior = diag(repmat(1/prior,1,D1));
    %For bayesian, if working on the diagonal then add the prior inverse
    HH = cellfun(@(~, ind) X * diag(Y(ind(1),:)' .* (ddirac(ind(1)-ind(2)) ...
        - Y(ind(2),:)')) * X' + (ind(1) == ind(2))*inv_prior, ...
        HH, index_mat_cell, 'UniformOutput', false);

    % Assemble final Hessian.
    %     for n = 1 : num_classes
    %         H_n = [];
    %         for m = 1 : num_classes
    %             H_n = [H_n HH{n,m}];
    %         end
    %         H1 = [H1; H_n];
    %     end

    %% Assemble using built in functions
    H = cell2mat(HH);
    disp(L);
end
