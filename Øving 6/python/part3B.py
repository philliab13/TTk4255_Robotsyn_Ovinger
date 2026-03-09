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

step = 10
idxs = np.arange(0, len(detections), step)

U_list, W_list = [], []
for i in idxs:
    w = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))
    U_list.append(u)
    W_list.append(w)

N = len(U_list)
angles0 = all_p[idxs, :].reshape(-1)

S  = make_jac_sparsity(N, 7, 39)


# Initial marker coords from heli_points (NOT platform corners)
X0 = quanser.heli_points[:3, :].reshape(-1, order="F")  # 21
X0[2]+=0.05
X0[5]+=0.05
X0[8]+=0.05

# Initial lengths consistent with your Part 1 model
lens0 = np.array([
    0.1145, 0.1145, 0.0,      # used as /2,/2,0 in your code
    0.0,    0.0,    0.325,    # base->hinge
    0.65,   0.0,   -0.080     # hinge->rotors: (-0.05) + (-0.03) = -0.08
], dtype=float)




ang_init=np.zeros(9)

p0 = np.hstack([ang_init,X0, lens0, angles0])
p0 = np.hstack([ang_init, X0, lens0, angles0])

K = 39          # Model A (21 marker coords + 9 lengths + 9 ang
M = 7
N = len(U_list)

S = make_jac_sparsity(N, M, K)

resfun = lambda p: quanser.residuals_multi_B(p, U_list, W_list)

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

idx = 0
alpha_hat = p_opt[idx:idx+9]; idx += 9

X_vec = p_opt[idx:idx+3*M]; idx += 3*M
X_hat = X_vec.reshape(3, M, order="F")

lens_hat = p_opt[idx:idx+9]; idx += 9

angles_hat = p_opt[idx:].reshape(N, 3)   # (N,3)

# residuals per image (N,14)
all_r2 = sol.fun.reshape(N, 2*M)
all_p2 = angles_hat
all_p_full = []
all_r_full = []

ang = all_p[0].copy()  # init from Part 1 angles (good starting point)

for i in range(len(detections)):
    w = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

    resfun_i = lambda ang: quanser.residuals_one_B(u, w, ang, alpha_hat, X_hat, lens_hat)

    ang = least_squares(resfun_i, x0=ang, method="lm").x  # 3 vars -> LM is fast

    all_p_full.append(ang)
    all_r_full.append(resfun_i(ang))

all_p_full = np.array(all_p_full)  # (N_full,3)
all_r_full = np.array(all_r_full)  # (N_full,14)

plot_all(all_p_full, all_r_full, detections, subtract_initial_offset=True)
plt.savefig("out_part3_fulltraj.png")
plt.show()



