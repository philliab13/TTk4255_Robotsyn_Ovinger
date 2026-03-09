from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from common import *  

def Rz(thz):return np.array([
    [np.cos(thz), -np.sin(thz), 0.0],
    [np.sin(thz),  np.cos(thz), 0.0],
    [0.0,          0.0,         1.0]
])

def Ry(thy): return np.array([
    [ np.cos(thy), 0.0, np.sin(thy)],
    [ 0.0,         1.0, 0.0        ],
    [-np.sin(thy), 0.0, np.cos(thy)]
])

def Rx(thyx): return np.array([
    [ 1, 0.0, 0.0],
    [ 0.0,         np.cos(thyx), -np.sin(thyx)        ],
    [0, np.sin(thyx), np.cos(thyx)]
])

K = np.loadtxt('data/heli_k.txt')
T_p_to_cam = np.loadtxt('data/platform_to_camera.txt')

#first 4.3
T_base_to_platform = np.eye(4, dtype=float)
t_p_t_b= np.array([0.1145/2, 0.1145/2, 0.0])

T_base_to_platform[:3, :3] = (Rz(np.deg2rad(11.6)))
T_base_to_platform[:3,  3] = t_p_t_b

#second 4.4
T_hinge_to_base=np.eye(4, dtype=float)
t_b_t_h= np.array([0, 0, 0.325])

T_hinge_to_base = np.eye(4)
T_hinge_to_base[:3, :3] = Ry(np.deg2rad(28.9))
T_hinge_to_base[:3,  3] = t_b_t_h

#4.5

t_h_t_a= np.array([0, 0, -0.05])

T_arm_to_hinge = np.eye(4)
T_arm_to_hinge[:3, :3] = np.eye(3)
T_arm_to_hinge[:3,  3] =t_h_t_a


#4.6

t_a_t_r= np.array([0.65, 0, -0.03])

T_rotor_to_arm = np.eye(4)
T_rotor_to_arm[:3, :3] = Rx(0)
T_rotor_to_arm[:3,  3] = t_a_t_r

X = np.loadtxt('data/heli_points.txt')
X=X.T

# Transform points into camera frame (homogeneous 4xN)
X_cam_rotor = T_p_to_cam@T_base_to_platform@T_hinge_to_base@T_arm_to_hinge@T_rotor_to_arm@  X[:, 3:]
X_cam_arm=T_p_to_cam@T_base_to_platform@T_hinge_to_base@T_arm_to_hinge@ X[:, :3]

# Project (your project() returns 2xN)
uv_arm=project(K,X_cam_arm)
uv = project(K, X_cam_rotor)
u_arm,v_arm=uv_arm[0,:], uv_arm[1,:]
u, v = uv[0, :], uv[1, :]

# Load and show image
img = np.array(Image.open('data/quanser.jpg'))

plt.figure(figsize=(8, 6))
plt.imshow(img)                 # origin='upper' by default => y points down

# 3) Zoom in by setting axis limits (left, right, bottom, top)
x1, y1, x2, y2 = 130, 300, 520, 600   # (left, top, right, bottom)
# Overlay points + frame
plt.scatter(u_arm, v_arm, c='r', s=40)
plt.scatter(u, v, c='r', s=40)
  # adjust scale if axes too big/small
T_cam_plat  = T_p_to_cam
T_cam_base  = T_p_to_cam @ T_base_to_platform
T_cam_hinge = T_cam_base @ T_hinge_to_base
T_cam_arm   = T_cam_hinge @ T_arm_to_hinge
T_cam_rotor = T_cam_arm @ T_rotor_to_arm

draw_frame(K, T_cam_plat,  scale=0.05)
draw_frame(K, T_cam_base,  scale=0.05)
draw_frame(K, T_cam_hinge, scale=0.05)
draw_frame(K, T_cam_arm,   scale=0.05)
draw_frame(K, T_cam_rotor, scale=0.05)

# Match axes to image pixel coords
h, w = img.shape[0], img.shape[1]
plt.xlim([0, w])
plt.ylim([h, 0])
plt.show()



# plt.xlim([x1, x2])
# plt.ylim([y2, y1])  # reversed because y goes downward in images

# plt.show()
