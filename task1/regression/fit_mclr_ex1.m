% Author: Stefan Stavrev 2013
close all;
clear all;

figure;

% Generate data for class 0.
I_0 = 10;
mu_0 = -2;
sigma_0 = 1.5;
class_0 = normrnd(mu_0, sigma_0, 1, I_0);
scatter(class_0, zeros(1,I_0),100,'fill','MarkerEdgeColor','k', ...
       'MarkerFaceColor','b','LineWidth',1);
hold on;

% Generate data for class 1.
I_1 = 10;
mu_1 = 0;
sigma_1 = 1.5;
class_1 = normrnd(mu_1, sigma_1, 1, I_1);
scatter(class_1, zeros(1,I_1),100,'fill','MarkerEdgeColor','k', ...
       'MarkerFaceColor','g','LineWidth',1);
   
% Generate data for class 2.
I_2 = 10;
mu_2 = 2;
sigma_2 = 1.5;
class_2 = normrnd(mu_2, sigma_2, 1, I_2);
scatter(class_2, zeros(1,I_2),100,'fill','MarkerEdgeColor','k', ...
       'MarkerFaceColor','r','LineWidth',1);
   
% Plot properties.
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
axis([-5,5,-0.1,1.1]);
set(gca,'XTick',[-5,0,5],'YTick',[0,1],'FontSize', 16);

% Prepare training data.
X = [class_0 class_1 class_2];
X = [ones(1,I_0+I_1+I_2); X];
w = [zeros(I_0,1); ones(I_1,1); 2*ones(I_2,1)] + 1;
X_test = -5:0.1:5;
X_test = [ones(1,size(X_test,2)); X_test];

% Fit a multi-class logistic regression model.
Predictions = fit_mclr (X, w, X_test, 3);

% Plot the predictions.
plot(-5:0.1:5, Predictions(1,:), 'LineWidth',2,'LineSmoothing','on','Color','b');
hold on;
plot(-5:0.1:5, Predictions(2,:), 'LineWidth',2,'LineSmoothing','on','Color','g');
hold on;
plot(-5:0.1:5, Predictions(3,:), 'LineWidth',2,'LineSmoothing','on','Color','r');
hold off;