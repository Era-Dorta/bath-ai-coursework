% Author: Stefan Stavrev 2013
clear all;
close all;

% Create a grid. Set granularity to 100/1000, for low/high quality plots.
granularity = 1000; 
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);

% Generate 2D data from normal distributions.
mu = [-3 2;      
      0 -3;      
      4 3];
sig = [0.5 0; 0 0.5];
X_data = [mvnrnd(mu(1,:), sig, 10);          
          mvnrnd(mu(2,:), sig, 10);          
          mvnrnd(mu(3,:), sig, 10);];
 
% Prepare the training input.
X_train = [ones(1,size(X_data,1)); X_data(:,1)'];
w = X_data(:,2);
nu = 0.0005;
X_test = [ones(1,granularity); domain];
kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 2);

% Fit a relevant vector regression model.
[mu_test, var_test, relevant] = fit_rvr (X_train, w, nu, X_test, kernel);

% Plot the predictive distribution.
figure;
Z = zeros(granularity,granularity);
for j = 1 : granularity
    mu = mu_test(j);
    for i = 1 : granularity
        ww = domain(i);
        Z(i,j) = normpdf(ww,mu,var_test(j));
    end
end
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
set(gca,'YTick',[a,b], 'XTick',[a,b], 'FontSize', 14);

% Plot the non-relevant data points.
hold on;
scatter(X_data(not(relevant),1), X_data(not(relevant),2), 50, 'fill',...
       'MarkerEdgeColor','w', 'MarkerFaceColor','b','LineWidth',2);
hold off;

% Plot the relevant data points.
hold on;
scatter(X_data(relevant,1), X_data(relevant,2), 200, 'fill',...
       'MarkerEdgeColor','w', 'MarkerFaceColor','b','LineWidth',2);
hold off;