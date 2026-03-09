import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.heli_points = np.loadtxt('data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

    def residuals(self, u, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        hat_u = project(self.K, np.hstack([p1, p2]))
        self.hat_u = hat_u # Save for use in draw()

        hat_u=np.array(hat_u)
        u=np.array(u)
        # TASK: Compute the vector of residuals.
        
        r=hat_u-u
        r_hor=r[0,:]
        r_vert=r[1,:]
    
        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        r = np.hstack((r_hor,r_vert))
        return r
    
    def residuals_multi(self, p, U_list, W_list):
        M = 7
        N = len(U_list)

        idx = 0

        # --- marker coords (21) ---
        X_vec = p[idx:idx + 3*M]; idx += 3*M
        X = X_vec.reshape(3, M, order="F")
        Xh = np.vstack([X, np.ones((1, M))])  # 4x7

        # --- lengths (5) ---
        l1, l2, l3, l4, l5 = p[idx:idx + 5]; idx += 5

        # --- angles (3N) ---
        angles = p[idx:].reshape(N, 3)

        r_all = []

        for i in range(N):
            yaw, pitch, roll = angles[i]
            u = np.asarray(U_list[i], dtype=float)                    # 2x7
            w = np.asarray(W_list[i], dtype=float).reshape(1, -1)     # 1x7

            # handle NaNs from missing detections
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

            # transforms (match Part 1 structure)
            base_to_platform = translate(l1/2, l1/2, 0.0) @ rotate_z(yaw)
            hinge_to_base    = translate(0.0, 0.0, l2) @ rotate_y(pitch)
            arm_to_hinge     = translate(0.0, 0.0, l3)
            rotors_to_arm    = translate(l4, 0.0, l5) @ rotate_x(roll)

            base_to_camera   = self.platform_to_camera @ base_to_platform
            hinge_to_camera  = base_to_camera @ hinge_to_base
            arm_to_camera    = hinge_to_camera @ arm_to_hinge
            rotors_to_camera = arm_to_camera @ rotors_to_arm

            p1 = arm_to_camera @ Xh[:, :3]
            p2 = rotors_to_camera @ Xh[:, 3:]
            hat_u = project(self.K, np.hstack([p1, p2]))
            hat_u = np.asarray(hat_u, dtype=float)                    # 2x7 expected

            diff = hat_u - u                                           # 2x7

            # force missing markers to contribute exactly 0
            diff[:, w.ravel() == 0] = 0.0

            # weight as sqrt(w) so sum(r^2) = sum(w * ||diff||^2)
            diff = diff * np.sqrt(w)

            r_all.append(np.hstack([diff[0, :], diff[1, :]]))

        return np.hstack(r_all)
    
    def residuals_multi_B(self, p, U_list, W_list):
        M = 7
        N = len(U_list)

        idx = 0
        alpha=p[idx:idx+9]; idx+=9

        # --- marker coords (21) ---
        X_vec = p[idx:idx + 3*M]; idx += 3*M
        X = X_vec.reshape(3, M, order="F")
        Xh = np.vstack([X, np.ones((1, M))])  # 4x7

        # --- lengths (5) ---
        l1, l2, l3, l4, l5,l6,l7,l8,l9 = p[idx:idx + 9]; idx += 9

        # --- angles (3N) ---
        angles = p[idx:].reshape(N, 3)

        r_all = []

        for i in range(N):
            yaw, pitch, roll = angles[i]
            u = np.asarray(U_list[i], dtype=float)                    # 2x7
            w = np.asarray(W_list[i], dtype=float).reshape(1, -1)     # 1x7

            # handle NaNs from missing detections
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

            # transforms (match Part 1 structure)
            T_1_to_plat = rotate_x(alpha[0])@rotate_y(alpha[1])@rotate_z(alpha[2])@translate(l1/2, l2/2, l3) @ rotate_z(yaw)
            T_2_to_1    = rotate_x(alpha[3])@rotate_y(alpha[4])@rotate_z(alpha[5])@translate(l4, l5, l6) @ rotate_y(pitch)
            T_3_to_2    = rotate_x(alpha[6])@rotate_y(alpha[7])@rotate_z(alpha[8])@translate(l7, l8, l9) @ rotate_x(roll)

            base_to_camera   = self.platform_to_camera @ T_1_to_plat
            hinge_to_camera  = base_to_camera @ T_2_to_1
            arm_to_camera    = hinge_to_camera @ T_3_to_2

            p1 = hinge_to_camera @ Xh[:, :3]
            p2 = arm_to_camera @ Xh[:, 3:]
            hat_u = project(self.K, np.hstack([p1, p2]))
            hat_u = np.asarray(hat_u, dtype=float)                    # 2x7 expected

            diff = hat_u - u                                           # 2x7

            # force missing markers to contribute exactly 0
            diff[:, w.ravel() == 0] = 0.0

            # weight as sqrt(w) so sum(r^2) = sum(w * ||diff||^2)
            diff = diff * np.sqrt(w)

            r_all.append(np.hstack([diff[0, :], diff[1, :]]))

        return np.hstack(r_all)
    def residuals_one_B(self, u, w, ang, alpha, X_hat, lens_hat):
        """
        One-image residual for Model B.
        u:        2x7 measured pixels (can contain NaNs)
        w:        length-7 weights (0/1 or general nonnegative)
        ang:      [yaw, pitch, roll] (radians)
        alpha:    length-9 (radians)
        X_hat:    3x7 marker coordinates
        lens_hat: length-9 translations [l1..l9]
        Returns:  1D residual vector length 14, ordered [x1..x7, y1..y7]
        """
        yaw, pitch, roll = ang

        u = np.asarray(u, dtype=float)
        if u.shape != (2, 7):
            u = u.T  # handle Nx2
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

        w = np.asarray(w, dtype=float).reshape(1, -1)  # 1x7

        X_hat = np.asarray(X_hat, dtype=float)
        if X_hat.shape != (3, 7):
            X_hat = X_hat.reshape(3, 7, order="F")
        Xh = np.vstack([X_hat, np.ones((1, 7))])  # 4x7

        l1, l2, l3, l4, l5, l6, l7, l8, l9 = np.asarray(lens_hat, dtype=float)
        alpha = np.asarray(alpha, dtype=float)

        # Model B transforms (same as in residuals_multi_B)
        T_1_to_plat = (
            rotate_x(alpha[0]) @ rotate_y(alpha[1]) @ rotate_z(alpha[2])
            @ translate(l1/2, l2/2, l3) @ rotate_z(yaw)
        )
        T_2_to_1 = (
            rotate_x(alpha[3]) @ rotate_y(alpha[4]) @ rotate_z(alpha[5])
            @ translate(l4, l5, l6) @ rotate_y(pitch)
        )
        T_3_to_2 = (
            rotate_x(alpha[6]) @ rotate_y(alpha[7]) @ rotate_z(alpha[8])
            @ translate(l7, l8, l9) @ rotate_x(roll)
        )

        base_to_camera  = self.platform_to_camera @ T_1_to_plat
        hinge_to_camera = base_to_camera @ T_2_to_1
        arm_to_camera   = hinge_to_camera @ T_3_to_2

        # first 3 markers use hinge_to_camera, last 4 use arm_to_camera (same split as your multi_B)
        p1 = hinge_to_camera @ Xh[:, :3]
        p2 = arm_to_camera @ Xh[:, 3:]
        hat_u = project(self.K, np.hstack([p1, p2]))
        hat_u = np.asarray(hat_u, dtype=float)
        if hat_u.shape != (2, 7):
            hat_u = hat_u.T

        diff = hat_u - u  # 2x7

        # zero-out missing markers
        diff[:, w.ravel() == 0] = 0.0

        # apply weights as sqrt(w)
        diff = diff * np.sqrt(w)

        return np.hstack([diff[0, :], diff[1, :]]).ravel()





    def draw(self, u, weights, image_number):
        I = plt.imread('quanser_image_sequence/data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*u[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.hat_u, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
