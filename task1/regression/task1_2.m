clear all;
close all;

load('data/Fonts_n_to_m.mat');

figure;
for i=1:size(X, 1)
    plotCharacter(X(i, :), 'b-');
end

figure;
for i=1:size(X, 1)
    plotCharacter(Y(i, :), 'b-');
end