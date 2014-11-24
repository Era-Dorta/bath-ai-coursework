% Author: Stefan Stavrev 2013

% The variance is expressed in terms of the standard deviation as
% var = sqrt(var)^2, to constrain var to take only positive values when
% this function is optimized in terms of var.
function f = fit_rvr_cost (var, K, w, H)
    I = length(w);
    H_inv = diag(1 ./ H);    
    covariance = K*H_inv*K + var*eye(I);
    f = mvnpdf (w, zeros(I,1), covariance);
    f = -log(f);
end