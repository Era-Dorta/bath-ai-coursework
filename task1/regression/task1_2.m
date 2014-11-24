clear all;
close all;

load('data/Fonts_n_to_m.mat');

n_train = size(trainingIndices, 1);
n_test = size(testIndices, 1);

X_train = [ones(1, n_train); X(trainingIndices, :)'];
X_test = [ones(1, n_test); X(testIndices, :)'];

colors = ['b','g','r','c','m','y','k', 'b','g','r','c','m','y','k'];

figure;
for i=1:size(X, 1)
    plotCharacter(X(i, :), strcat(colors(i),'-'));
end

figure;
for i=1:size(X, 1)
    plotCharacter(Y(i, :), strcat(colors(i),'-'));
end