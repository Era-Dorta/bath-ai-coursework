function [ cluster_mu, cluster_assig ] = kmeans( X, K )
    dimensionality = size(X, 2);
    I = size(X, 1);

    % Compute overall mean
    overall_mu = sum(X) / I;
    
    % Compute overall covariance
    overall_covar = zeros (dimensionality, dimensionality);
    for i = 1 : I
        mat = X (i,:) - overall_mu;
        mat = mat' * mat;
        overall_covar = overall_covar + mat;
    end
    overall_covar = overall_covar ./ I;
    
    % Set first K points to each cluster and the rest randomly
    cluster_assig = [1:K, randi(K,1,I - K)];
       
    % Initialize each cluster randomly
    cluster_mu = mvnrnd(overall_mu, overall_covar, K);
    
    distances = zeros(I, K);
    
    not_change = true;
    while not_change
        not_change = false;
        % Compute distance from data points to cluster means
        for i=1:I
            for k=1:K
                distances(i,k) = (X(i,:) - cluster_mu(k,:)) * (X(i,:) - cluster_mu(k,:))';
            end
            % Update cluster assignments based on closest cluster
            [~, new_cluster] = min(distances(i,:));
            % If the cluster is going to get empty then don't update, as the
            % distance will be zero when it gets recalculated on the next
            % step
            if new_cluster ~= cluster_assig(i) && length(find(cluster_assig == cluster_assig(i))) > 1
                cluster_assig(i) = new_cluster;
                not_change = true;
            end
        end
        
        % Update cluster means from data that was assigned to this cluster
        for k=1:K
            cluster_ind = find(cluster_assig == k);
            cluster_mu(k,:) = sum(X(cluster_ind, :)) / length(cluster_ind);
        end
    end
end

