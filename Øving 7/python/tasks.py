import numpy as np

def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    #Build A
    n = XY.shape[1]

    # Build A matrix (2n x 9)
    A = np.zeros((2 * n, 9))

    for i in range(n):
        
        X_i, Y_i = XY[0:2, i]
        x_i, y_i = xy[0:2, i]

        A[2 * i] = [X_i, Y_i, 1, 0, 0, 0, -X_i * x_i, -Y_i * x_i, -x_i]
        A[2 * i + 1] = [0, 0, 0, X_i, Y_i, 1, -X_i * y_i, -Y_i * y_i, -y_i]

    U,s,VT=np.linalg.svd(A)
    h=VT.T[:,-1]
    H = h.reshape(3, 3)

    return H


def decompose_H(H):
    # Tip: Use np.linalg.norm to compute the Euclidean length
    r1 = H[:, 0].copy()
    r2 = H[:, 1].copy()
    tx, ty, tz=H[:,2].copy()
    k=np.linalg.norm(r1)
    
    t=np.array([tx,ty,tz]).T
    r1/=k
    r2/=k
    t/=k
    r3=np.linalg.cross(r1,r2)

    R = np.column_stack((r1, r2, r3))
    # R=closest_rotation_matrix(R)

    # e_det=np.abs(1-np.linalg.det(R))
    # print(f"Error in determinant: {e_det}")
    # e_inverse=np.sum(R.T-np.linalg.inv(R))/9
    # print(f"Error in transpose=inverse: {e_inverse}")

    T1 = np.eye(4)
    T1[:3, :3] = R
    T1[:3, 3] = t

    T2 = np.eye(4)
    T2[:3, :3] = R.copy()
    T2[:3, 0] *= -1
    T2[:3, 1] *= -1
    T2[:3, 3] = -t
    return T1, T2

def closest_rotation_matrix(Q):
    U,s,VT=np.linalg.svd(Q)
    R=U@VT
    return R
