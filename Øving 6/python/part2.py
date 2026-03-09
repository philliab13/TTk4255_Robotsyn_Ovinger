import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.optimize import least_squares

K = np.loadtxt('data/K.txt')
u = np.loadtxt('data/platform_corners_image.txt')
X = np.loadtxt('data/platform_corners_metric.txt')
I = plt.imread('quanser_image_sequence/data/video0000.jpg')
print(u[:,:3])
u=u[:,:3]
X = X[:, :3]

# This is just an example. Replace these two lines
# with your own code.
R0=np.eye(4)
R0=rotate_x(np.deg2rad(30))@rotate_y(np.deg2rad(40))

#LM
def T_of(p):
    tx, ty, tz, roll, pitch, yaw = p
    return translate(tx, ty, tz) @ rotate_x(roll) @ rotate_y(pitch) @ rotate_z(yaw) @ R0


def residuals_T(p):
    T = T_of(p)
    uhat = project(K, T @ X)
    uhat=uhat[:,:3]

    diff = uhat - u  # 2xN
    return np.hstack([diff[0, :], diff[1, :]]).ravel()

p0 = np.array([2, 4, 0.8, np.deg2rad(180), np.deg2rad(70), np.deg2rad(30)], dtype=float)

sol = least_squares(residuals_T, x0=p0, method='lm')
p_hat = sol.x
T_hat = T_of(p_hat)

hat_u = project(K, T_hat@X)
hat_u=hat_u[:,:3]
print(T_hat)



reprojection_errors = np.linalg.norm(u - hat_u, axis=0)
print('Reprojection errors:')
for e in reprojection_errors:
    print('%.05f px' % e)

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(hat_u[0,:], hat_u[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw the axes of a coordinate frame
draw_frame(K, T_hat, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
# plt.xlim([200, 500])
# plt.ylim([600, 350])

plt.savefig('out_part2.png')
plt.show()
