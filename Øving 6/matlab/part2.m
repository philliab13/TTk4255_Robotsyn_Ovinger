clear; clc;

K = load('../data/K.txt');
u = load('../data/platform_corners_image.txt');
X = load('../data/platform_corners_metric.txt');
I = imread('../data/video0000.jpg');

% This is just an example. Replace these two lines
% with your own code.
hat_T = translate(0.0, 0.0, 1.0);
hat_u = project(K, hat_T*X);

fprintf('Reprojection errors:\n')
reprojection_errors = vecnorm(u - hat_u, 2, 1);
for i=1:length(reprojection_errors)
    fprintf('%.05f px\n', reprojection_errors(i));
end

figure; clf;
imshow(I);
hold on;
scatter(u(1,:), u(2,:), 100, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'LineWidth', 1.5);
scatter(hat_u(1,:), hat_u(2,:), 30, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');

% Tip: Draw the axes of a coordinate frame
draw_frame(K, hat_T, 0.05, true);

% Tip: To zoom in on the platform:
% xlim([200, 500]);
% ylim([350, 600]);

legend('Detected', 'Predicted');
