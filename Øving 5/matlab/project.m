function uv = project(X, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    % Calculate the projected pixel coordinates of a single 3x1 point
    % in the pinhole camera model with Brown-Conrady distortion.

    x = X(1)/X(3);
    y = X(2)/X(3);
    r2 = x*x + y*y;
    r4 = r2*r2;
    r6 = r4*r2;
    dr = (k1*r2 + k2*r4 + k3*r6);
    dx = dr*x + 2*p1*x*y + p2*(r2 + 2*x*x);
    dy = dr*y + p1*(r2 + 2*y*y) + 2*p2*x*y;
    u = cx + fx*(x + dx);
    v = cy + fy*(y + dy);
    uv = [u ; v];
end
