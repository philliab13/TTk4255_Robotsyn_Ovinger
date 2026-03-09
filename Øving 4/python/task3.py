from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from common import *  # assumes you have project() + draw_frame()

K = np.loadtxt('data/heli_k.txt')
T_p_to_cam = np.loadtxt('data/platform_to_camera.txt')

X = np.array([
    [0, .1145, .1145, 0],
    [0, 0,     0.1145, .1145],
    [0, 0,     0,     0],
    [1, 1,     1,     1]
], dtype=float)

# Transform points into camera frame (homogeneous 4xN)
X_cam = T_p_to_cam @ X

# Project (your project() returns 2xN)
uv = project(K, X_cam)
u, v = uv[0, :], uv[1, :]

# Load and show image
img = np.array(Image.open('data/quanser.jpg'))

plt.figure(figsize=(8, 6))
plt.imshow(img)                 # origin='upper' by default => y points down

# 3) Zoom in by setting axis limits (left, right, bottom, top)
x1, y1, x2, y2 = 130, 300, 520, 600   # (left, top, right, bottom)
# Overlay points + frame
plt.scatter(u, v, c='r', s=40)
draw_frame(K, T_p_to_cam, scale=1.0)  # adjust scale if axes too big/small

# Match axes to image pixel coords
h, w = img.shape[0], img.shape[1]




plt.xlim([x1, x2])
plt.ylim([y2, y1])  # reversed because y goes downward in images

plt.show()
