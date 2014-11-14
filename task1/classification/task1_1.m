clear all;
close all;

data_path = {'data/MNIST_Data.mat', 'data/ETH-80_HoG_Data.mat'};
prior = [1, 10, 100, 1000];
initial_phi = [0.1, -1, 1, 2.5; 0.125, -10, -1, 10];

for i=1:length(data_path)
    for j=1:length(prior)
        for k=1:length(initial_phi)
            hits = fit_bmclr_ex1(i, data_path{i}, prior(j), initial_phi(i, k));
            fprintf('Data %s, prior %2.2f, initial phi %2.2f,  hits: %2.2f%%\n', ...
            data_path{i}, prior(j), initial_phi(i, k), hits);
        end
    end
end