close all;
clear all;

% Generate 2D data from 2 normal distributions.
mu1 = [1 2];
sig1 = [2 0; 0 .5];
mu2 = [1 5];
sig2 = [1 0; 0 1];
X = [mvnrnd(mu1,sig1,5000); mvnrnd(mu2,sig2,5000)];

% Plot the data.
scatter(X(:,1), X(:,2), 10, '.');
axis([-6,6,-1,9]);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca,'YTick',[-1,9], 'XTick',[-6,6]);
title('(a)', 'FontSize', 14);

% Fit MoG using our function fit_mog.
[lambda, mu, sig] = fit_mog (X, 2, 0.01);

% Fit MoG using Matlab's function gmdistribution.fit, for comparison.
options = statset('Display','final');
obj = gmdistribution.fit(X,2,'Options',options);

% Print and compare results from ours and Matlab's function. The results
% should be exactly the same, although not necessarily in the same order.
% For example mu(1,:) may be equal to obj.mu(2,:), and mu(2,:) to
% obj.mu(1,:). This order is irelevant however.
disp(obj.PComponents);
disp(lambda');

disp(obj.mu);
disp(mu);

disp(obj.Sigma);
disp(sig{1});
disp(sig{2});

% Plot the mixture of Gaussians as contour plot.
% Create the x-y matrix.
[XX,YY] = meshgrid (-10:0.01:10,-10:0.01:10);
x = XX(:);
y = YY(:);
x_y_matrix = [x y];
n = size(XX,1);

% Compute the first Gaussian as a matrix.
temp1 = mvnpdf (x_y_matrix, mu(1,:), sig{1});
gaussian1 = reshape (temp1, n, n);

% Compute the second Gaussian as a matrix.
temp2 = mvnpdf (x_y_matrix, mu(2,:), sig{2});
gaussian2 = reshape (temp2, n, n);

% Now combine the two Gaussians with the corresponding weights in lambda,
% to obtain the final mixture of Gaussians.
mog = lambda(1)*gaussian1 + lambda(2)*gaussian2;

% Create the contour plot.
figure;
scatter(X(:,1), X(:,2), 10, '.');
axis([-6,6,-1,9]);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca,'YTick',[-1,9], 'XTick',[-6,6]);
title('(b)', 'FontSize', 14);
hold on;
contour(XX,YY,mog);
hold off;