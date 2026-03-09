from scipy.optimize import least_squares
import numpy as np
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
angles0 = np.zeros(3*N, dtype=float)

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
print(sol.x)
