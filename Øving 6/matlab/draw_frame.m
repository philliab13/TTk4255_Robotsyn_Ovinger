function draw_frame(K, T, scale, labels)
    % Visualize the coordinate frame axes of the 4x4 object-to-camera
    % matrix T using the 3x3 intrinsic matrix K.
    %
    % Control the length of the axes by specifying the scale argument.

    if ~exist('labels', 'var')
        labels = false;
    end

    uO = project(K, T*[0 0 0 1]');
    uX = project(K, T*[scale 0 0 1]');
    uY = project(K, T*[0 scale 0 1]');
    uZ = project(K, T*[0 0 scale 1]');
    plot([uO(1) uX(1)], [uO(2) uX(2)], 'color', '#cc4422', 'linewidth', 2); % X-axis
    plot([uO(1) uY(1)], [uO(2) uY(2)], 'color', '#11ff33', 'linewidth', 2); % Y-axis
    plot([uO(1) uZ(1)], [uO(2) uZ(2)], 'color', '#3366ff', 'linewidth', 2); % Z-axis
    if labels
        text(uX(1), uX(2), 'X', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        text(uY(1), uY(2), 'Y', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        text(uZ(1), uZ(2), 'Z', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
