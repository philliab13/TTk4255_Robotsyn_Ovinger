import numpy as np
import matplotlib.pyplot as plt
from common import *

# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt('data/task2K.txt')
X = np.loadtxt('data/task3points.txt')

# Task 2.2: Implement the project function

def T_zm15_ym45(order="RzRy"):
    """
    Build 4x4 homogeneous transform with:
      - rotation about z by -15 deg and about y by -45 deg
      - translation +6 in z direction
    order:
      "RzRy" -> R = Rz(-15) @ Ry(-45)
      "RyRz" -> R = Ry(-45) @ Rz(-15)
    """
    thz = np.deg2rad(15.0)
    thy = np.deg2rad(45.0)
    thyx=np.deg2rad(15.0)

    Rz = np.array([
        [np.cos(thz), -np.sin(thz), 0.0],
        [np.sin(thz),  np.cos(thz), 0.0],
        [0.0,          0.0,         1.0]
    ])

    Ry = np.array([
        [ np.cos(thy), 0.0, np.sin(thy)],
        [ 0.0,         1.0, 0.0        ],
        [-np.sin(thy), 0.0, np.cos(thy)]
    ])

    Rx=np.array([
        [ 1, 0.0, 0.0],
        [ 0.0,         np.cos(thyx), -np.sin(thyx)        ],
        [0, np.sin(thyx), np.cos(thyx)]
    ])

    if order == "RzRy":
        R = Rz @ Ry
    elif order == "RyRz":
        R = Ry @ Rz
    elif order=="RxRy":
        R=Rx@Ry

    else:
        raise ValueError('order must be "RzRy" or "RyRz"')

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = np.array([0.0, 0.0, 6.0])  # translation in z
    return T

# Example:
T = T_zm15_ym45(order="RxRy")
print(T.shape)
print(X.shape)
X_new=T@X


draw_frame(K,T)
u,v = project(K, X_new)


# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400

#
# Figure for Task 2.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
plt.show()
