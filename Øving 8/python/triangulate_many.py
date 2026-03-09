import numpy as np
import pandas as pd

def triangulate_many(u1, u2, P1, P2):
    """
    Arguments
        u: Image coordinates in image 1 and 2
        P: Projection matrix K[R t] for image 1 and 2
    Returns
        X: Homogeneous coordinates of 3D points (shape 4 x n)
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    n = u1.shape[1]

    x1 = u1[0, :] / u1[2, :]
    y1 = u1[1, :] / u1[2, :]
    x2 = u2[0, :] / u2[2, :]
    y2 = u2[1, :] / u2[2, :]

    A = np.zeros((n, 4, 4), dtype=float)
    A[:, 0, :] = x1[:, None] * P1[2, :] - P1[0, :] #Eq 22
    A[:, 1, :] = y1[:, None] * P1[2, :] - P1[1, :]
    A[:, 2, :] = x2[:, None] * P2[2, :] - P2[0, :]
    A[:, 3, :] = y2[:, None] * P2[2, :] - P2[1, :]

    _, _, Vt = np.linalg.svd(A)

    X = Vt[:, -1, :]   # shape (n, 4)

    X = X / X[:, 3:4]

 
    return X.T

