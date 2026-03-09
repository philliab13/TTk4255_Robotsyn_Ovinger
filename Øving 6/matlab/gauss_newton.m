function p = gauss_newton(resfun, p0, step_size, num_steps, finite_difference_epsilon)
    jacfun = @(p) compute_jacobian(resfun, p, finite_difference_epsilon)
    r = resfun(p0);
    J = jacfun(p0);
    p = p0;
    for iteration=1:num_steps
        A = J'*J;
        b = -J'*r;
        d = A \ b;
        p = p + step_size*d;
        r = resfun(p);
        J = jacfun(p);
    end
end
