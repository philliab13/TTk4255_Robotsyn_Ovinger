from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from common import *

# Note: the sample image is naturally grayscale
I = rgb_to_gray(im2double(plt.imread('data/calibration.jpg')))

###########################################
#
# Task 3.1: Compute the Harris-Stephens measure
#
###########################################
sigma_D = 1
sigma_I = 3
alpha = 0.06
Ix, Iy, Im = derivative_of_gaussian(I, sigma_D)
Sxx = ndimage.gaussian_filter(Ix * Ix, sigma=sigma_I, mode='reflect')
Sxy = ndimage.gaussian_filter(Ix * Iy, sigma=sigma_I, mode='reflect')
Syy = ndimage.gaussian_filter(Iy * Iy, sigma=sigma_I, mode='reflect')

response = (Sxx * Syy - Sxy * Sxy) - alpha * \
    (Sxx + Syy) ** 2  # right side of eq2


###########################################
#
# Task 3.4: Extract local maxima
#
###########################################


row, col = extract_local_maxima(response, 0.001)
corners_y = row  # Placeholder
corners_x = col  # Placeholder

###########################################
#
# Figure 3.1: Display Harris-Stephens corner strength
#
###########################################
plt.figure(figsize=(13, 5))
plt.imshow(response)
plt.colorbar(label='Corner strength')
plt.tight_layout()
# plt.savefig('out_corner_strength.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

###########################################
#
# Figure 3.4: Display extracted corners
#
###########################################
plt.figure(figsize=(10, 5))
plt.imshow(I, cmap='gray')
plt.scatter(corners_x, corners_y, linewidths=1,
            edgecolor='black', color='yellow', s=9)
plt.tight_layout()
# plt.savefig('out_corners.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

plt.show()
