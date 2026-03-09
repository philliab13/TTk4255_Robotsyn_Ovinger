function e = epipolar_distance(F, u1, u2)
    % F should be the fundamental matrix (use F_from_E).
    % u1, u2 should be arrays of size 3 x n containing
    % homogeneous pixel coordinates.

    n = size(u1, 2);
    e = zeros(1, n); % Placeholder, replace with your implementation
end
