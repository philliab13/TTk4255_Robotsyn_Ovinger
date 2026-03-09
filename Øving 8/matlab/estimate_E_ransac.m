function [E,inliers] = estimate_E_ransac(B1, B2, K, distance_threshold, num_trials)

    % Tip: The following snippet extracts a random subset of 8
    % correspondences (w/o replacement) and estimates E using them.
    %   sample = randperm(size(B1, 2), 8);
    %   E = estimate_E(B1(:,sample), B2(:,sample));

end
