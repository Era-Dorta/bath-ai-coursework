% Author: Stefan Stavrev 2013
clear all;
close all;

load('data/digitsData.mat');

% Get the indices of all the samples of classes 0 and 1
X01Indices = find(Y <= 1);

newY = zeros(length(X01Indices), 1);

% Save instances of classes 0 and 1 in separated variables
for i=1:length(X01Indices)
    newX(i, :) = X(X01Indices(i), :);
    newY(i) = Y(X01Indices(i));
end

n_train = 10;
n_test = 10;

% Pick the first n_train samples for training
X_train = newX(1:n_train, :);
Y_train = newY(1:n_train);

% And the next n_test samples for testing
X_test = newX(n_train + 1:n_train + n_test, :);
Y_test = newY(n_train + 1:n_train + n_test);


% Format the data for fit_blogr
data_n_dims = size(X_train, 2);

X_train_blogr = [ones(1,size(X_train,1)); X_train'];
X_test_blogr = [ones(1,size(X_test,1)); X_test'];
w = Y_train;


% Create a grid. Set granularity to 100/1000, for low/high quality plots.
% granularity = 100; 
% a = -5;
% b = 5;
% domain = linspace (a, b, granularity);
% [X,Y] = meshgrid (domain, domain);
% x = X(:);
% y = Y(:);
% n = size(X,1);

% Generate 2D data from normal distributions.
% mu = [-1 2.5;
%       1 -2.5];
% sig = [0.5 0; 0 0.5];
% points_per_class = 20;
% X_data = [mvnrnd(mu(1,:), sig, points_per_class);          
%           mvnrnd(mu(2,:), sig, points_per_class);];
%       
% % Prepare the training input.
% X_train = [ones(1,size(X_data,1)); X_data'];
% w = [zeros(points_per_class,1); ones(points_per_class,1)];
% var_prior = 6;
% X_test = [ones(1,granularity*granularity); x'; y'];

% Fit a Bayesian logistic regression model.
%initial_phi = [1; 1; 1];

var_prior = 6;
initial_phi = 10000 * ones(data_n_dims + 1, 1);
[predictions, phi] = fit_blogr (X_train_blogr, w, var_prior, X_test_blogr, initial_phi);

%disp(predictions);
%disp(phi(1:10));


% Plot the results.
% figure;
% Z = reshape(predictions,n,n);
% pcolor(X,Y,real(Z));
% shading interp;
% colormap(hot(4096));
% set(gca,'YDir','normal');
% axis([a,b,a,b]);
% set(gca,'YTick',[a,0,b], 'XTick',[a,0,b], 'FontSize', 16);
% xlabel('x', 'FontSize', 16);
% ylabel('y', 'FontSize', 16);
% title('(b)', 'FontSize', 16);
% 
% % Plot the data points on top.
% hold on;
% selector = 1:points_per_class;
% scatter(X_data(selector,1), X_data(selector,2), 100, 'fill',...
%        'MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1);
% 
% hold on;
% selector = points_per_class+1 : 2*points_per_class;
% scatter(X_data(selector,1), X_data(selector,2), 100, 'fill',...
%        'MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1);
% 
% hold on;
% decision_boundary = -(phi(1) + phi(2)*domain) / phi(3);
% plot(domain,decision_boundary,'LineWidth',2,'LineSmoothing','on',...
%     'Color','c');
% hold off;