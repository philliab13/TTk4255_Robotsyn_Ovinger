clear; clc;

% Image to load data for (must be in [0, 350])
image_number = 0;

% Tip:
% "u" is a 2x7 array of detected marker locations.
% It is the same size in every image, but some of its
% entries may be invalid if the corresponding markers were
% not detected. Which entries are valid is encoded in
% the "weights" array, which is a 1D array of length 7.
n = 7; % Number of markers
detections = load('../data/detections.txt');
weights = detections(image_number + 1, 1:3:end);
u = [detections(image_number + 1, 2:3:end) ;
     detections(image_number + 1, 3:3:end) ];

quanser = Quanser;

% Tip:
% Many optimization packages for Matlab expect you to provide a
% callable function that computes the residuals, and optionally
% the Jacobian, at a given parameter vector. The provided Gauss-Newton
% implementation also follows this practice. However, because the
% "residuals" method takes arguments other than the parameters, you
% must first define a "lambda function wrapper" that takes only a
% single argument (the parameter vector), and likewise for computing
% the Jacobian. This can be done like this:
resfun = @(p) quanser.residuals(u, weights, p(1), p(2), p(3));

% Tip:
% These parameter values (yaw, pitch, roll) are close to the optimal
% estimates for image 0.
p = [11.6, 28.9, 0.0]*pi/180;

%
% Task: Call gauss_newton
%

fprintf('Residuals on image %d:\n', image_number);
r = resfun(p)

% Tip:
% This snippet produces the requested outputs for the second half
% of Task 1.3.
reprojection_errors = vecnorm([r(1:n)' ; r(n+1:2*n)']);
fprintf('Reprojection errors at solution:\n')
for i=1:length(reprojection_errors)
    fprintf('Marker %d: %5.02f px\n', i, reprojection_errors(i));
end
fprintf('Average:  %5.02f px\n', mean(reprojection_errors));
fprintf('Median:   %5.02f px\n', median(reprojection_errors));
quanser.draw(u, weights, image_number);
