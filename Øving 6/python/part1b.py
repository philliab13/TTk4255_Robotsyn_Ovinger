# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
def make_jac_sparsity(N, M, K):
    m = 2 * M * N          # residual dimension
    n = K + 3 * N          # variable dimension
    S = lil_matrix((m, n), dtype=int)

    for i in range(N):
        r0 = 2 * M * i
        r1 = 2 * M * (i + 1)

        # all residuals in image i depend on all kinematic params
        S[r0:r1, 0:K] = 1

        # ...and on that image's own yaw/pitch/roll
        c0 = K + 3 * i
        c1 = K + 3 * (i + 1)
        S[r0:r1, c0:c1] = 1

    return S

detections = np.loadtxt('data/detections.txt')
quanser = Quanser()

p = np.array([0.0, 0.0, 0.0])
all_r = []
all_p = []
for i in range(len(detections)):
    weights = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

    # Tip: Lambda functions can be defined inside a for-loop, defining
    # a different function in each iteration. Here we pass in the current
    # image's "u" and "weights".
    resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])

    # Tip: Use the previous image's parameter estimate as initialization
    p = least_squares(resfun, x0=p, method='lm').x

    # Collect residuals and parameter estimate for plotting later
    all_r.append(resfun(p))
    all_p.append(p)
all_p = np.array(all_p)
all_r = np.array(all_r)

# Tip: This saves the estimated angles to a txt file.
# This can be useful for Part 3.
# np.savetxt('trajectory_from_part1.txt', all_p)
# It can be loaded as
# all_p = np.loadtxt('trajectory_from_part1.txt')

# Tip: See comment in plot_all.py regarding the last argument.
plot_all(all_p, all_r, detections, subtract_initial_offset=True)
plt.savefig('out_part1b.png')
plt.show()

#Part 3
detections = np.loadtxt('data/detections.txt')
quanser = Quanser()

# Build lists of u_i and w_i for all images (or a subset)
U_list, W_list = [], []
for i in range(len(detections)):
    w = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))
    U_list.append(u)
    W_list.append(w)

N = len(U_list)
M = 7

# Initial marker coords from heli_points (NOT platform corners)
X0 = quanser.heli_points[:3, :].reshape(-1, order="F")  # 21

# Initial lengths consistent with your Part 1 model
lens0 = np.array([0.1145, 0.325, -0.050, 0.65, -0.030], dtype=float)

# Initial angles: use Part 1 trajectory if you have it; else zeros
angles0 = all_p.reshape(-1)

p0 = np.hstack([X0, lens0, angles0])

K = 26          # Model A (21 marker coords + 5 lengths)
M = 7
N = len(U_list)

S = make_jac_sparsity(N, M, K)

resfun = lambda p: quanser.residuals_multi(p, U_list, W_list)

sol = least_squares(
    resfun,
    x0=p0,
    method="trf",          # IMPORTANT: not 'lm'
    jac="2-point",         # finite-diff Jacobian
    jac_sparsity=S,
    x_scale="jac",
    verbose=2
)

print(sol.success, sol.message)
print("final RMSE px:", np.sqrt(np.mean(sol.fun**2)))


p_opt = sol.x

M = 7
N = len(U_list)
Kkin = 26  # 21 marker coords + 5 lengths

# Unpack marker coords (not needed for plot_all, but useful)
idx = 0
X_vec = p_opt[idx:idx + 3*M]; idx += 3*M
X_hat = X_vec.reshape(3, M, order="F")

# Unpack lengths
lens_hat = p_opt[idx:idx + 5]; idx += 5

# Unpack angles for all images
angles_hat = p_opt[idx:].reshape(N, 3)   # (N,3) -> yaw,pitch,roll

# Now compute residuals per image using the optimized shared kinematics
all_r2 = []
for i in range(N):
    u_i = U_list[i]
    w_i = W_list[i]

    # Build a per-image p vector for residuals_multi: [X_hat(21), lens_hat(5), angles_i(3)]
    p_i = np.hstack([X_hat.reshape(-1, order="F"), lens_hat, angles_hat[i]])

    # residuals_multi expects lists; give it one-image lists
    r_i = quanser.residuals_multi(p_i, [u_i], [w_i])   # returns length 14
    all_r2.append(r_i)

all_r2 = np.array(all_r2)      # (N,14)
all_p2 = angles_hat            # (N,3)

plot_all(all_p2, all_r2, detections, subtract_initial_offset=True)
plt.savefig('out_part3_traj.png')
plt.show()
print(p_opt)
