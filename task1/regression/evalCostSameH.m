function [ testCost ] = evalCostSameH( var, K, w, H, D, nu )
% Computes a combined cost function based on the rvr cost of each dimension
testCost = 0;
for loopCount = 1:D
    testCost = testCost + fit_rvr_cost_nu (var(loopCount), K, w(:,loopCount), H(:,1),nu);
end

end

