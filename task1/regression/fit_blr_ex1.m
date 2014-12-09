% Create one common grid for all 2D functions to be computed later.
% Set granularity to 100/1000, for low/high quality plots.
granularity = 1000; 
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
x_y_matrix = [x y];
n = size(X,1);

% Compute the prior 2D normal distribution over phi.
mu_1 = [0 0];
var_prior = 6;
covariance_1 = var_prior * eye(2);
mvnpdf_1 = mvnpdf (x_y_matrix, mu_1, covariance_1);
mvnpdf_1 = reshape (mvnpdf_1, n, n);

% Plot the prior 2D normal distribution over phi.
subplot(1,2,1);
pcolor(X,Y,mvnpdf_1);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(a)', 'FontSize', 14);

% Generate the training and test data.
X_train = [1 -4;
           1 -1;
           1 -1;           
           1 0;
           1 1;
           1 3.5]';
 w = [4.5 3 2 2.5 2.5 0]';
 X_test = linspace (a, b, granularity);
 X_test = [ones(1,granularity); X_test];

% Fit Bayesian linear regression model.
[mu_test, var_test, var, A_inv] = fit_blr (X_train, w, var_prior, X_test);

% Compute the posterior 2D normal distribution over phi.
mu_2 = ((A_inv*X_train*w) / var)';
covariance_2 = A_inv;
mvnpdf_2 = mvnpdf (x_y_matrix, mu_2, covariance_2);
mvnpdf_2 = reshape (mvnpdf_2, n, n);

% Plot the posterior 2D normal distribution over phi.
subplot(1,2,2);
pcolor(X,Y,mvnpdf_2);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(b)', 'FontSize', 14);

% Sample from the posterior and plot.
phi_samples = mvnrnd(mu_2, covariance_2, 3);

% Plot the 3 samples.

% Plot sample 1.
figure;
subplot(3,1,1);
XX = [ones(granularity,1) domain'];
temp = XX * phi_samples(1,:)';
Z = zeros(granularity,granularity);
for j = 1 : granularity
    mu = temp(j);
    for i = 1 : granularity
        ww = domain(i);
        Z(i,j) = normpdf(ww,mu,var);
    end
end
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[]);
title('(a)', 'FontSize', 14);

% Plot sample 2.
subplot(3,1,2);
temp = XX * phi_samples(2,:)';
for j = 1 : granularity
    mu = temp(j);
    for i = 1 : granularity
        ww = domain(i);
        Z(i,j) = normpdf(ww,mu,var);
    end
end
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[]);
title('(b)', 'FontSize', 14);

% Plot sample 3.
subplot(3,1,3);
temp = XX * phi_samples(3,:)';
for j = 1 : granularity
    mu = temp(j);
    for i = 1 : granularity
        ww = domain(i);
        Z(i,j) = normpdf(ww,mu,var);
    end
end
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(c)', 'FontSize', 14);

% Plot the main example for Bayesian linear regression.
figure;
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
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(d)', 'FontSize', 14);

% Plot the training data points.
hold on;
scatter(X_train(2,:),w',120,'fill','MarkerEdgeColor','w',...
       'MarkerFaceColor','b','LineWidth',2);
hold off;