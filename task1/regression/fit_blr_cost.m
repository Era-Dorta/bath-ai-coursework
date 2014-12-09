% The variance is expressed in terms of the standard deviation as
% var = sqrt(var)^2, to constrain var to take only positive values when
% this function is optimized in terms of var.
function f = fit_blr_cost (var, X, w, var_prior)
    I = size(X,2);
    covariance = var_prior*(X'*X) + (sqrt(var)^2)*eye(I);
    f = mvnpdf (w, zeros(I,1), covariance);
    f = -log(f);
end