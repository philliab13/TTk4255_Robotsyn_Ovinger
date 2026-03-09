function uv = project(K, X)
    % Computes the pinhole projection of a 3xN array of 3D points X
    % using the camera intrinsic matrix K. Returns the dehomogenized
    % pixel coordinates as an array of size 2xN.
    uvw = K*X(1:3,:);
    uv = [uvw(1,:)./uvw(3,:) ; uvw(2,:)./uvw(3,:)];
end
