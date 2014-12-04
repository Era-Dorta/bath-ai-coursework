function [ testCost ] = evalCost( var, K, w, H, D )
% Computes a combined cost function based on the rvr cost of each dimension
testCost = 0;
for loopCount = 1:D
    testCost = testCost + fit_rvr_cost (var(loopCount), K, w(:,loopCount), H(:,loopCount));
end

end

