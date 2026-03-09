import numpy as np
import pandas as pd

#This is one back-projection error, columns are the vectors
    # print(B1[:,0]) -> B2
    # print(B1[:,1]) -> C2
    # print(B1[:,0]) -> D2, of the first point correspondance
    # print(B1[0,:]) -> All of the B1 points

def estimate_E(B1, B2):
    # Bi: Array of size 3 x n containing back-projection vectors in image i.
    b1=np.array(B1[0,:])
    c1=np.array(B1[1,:])
    d1=np.array(B1[2,:])

    b2=np.array(B2[0,:])
    c2=np.array(B2[1,:])
    d2=np.array(B2[2,:])


    b2b1=b2*b1
    b2c1=b2*c1
    b2d1=b2*d1
    c2b1=c2*b1
    c2c1=c2*c1
    c2d1=c2*d1
    d2b1=d2*b1
    d2c1=d2*c1
    d2d1=d2*d1

    A=np.column_stack((b2b1,b2c1,b2d1,c2b1,c2c1,c2d1,d2b1,d2c1,d2d1))

    _, _, Vh = np.linalg.svd(A)
    E = Vh[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(E)
    
    S = [1, 1, 0] # Rank 2 constraint
    E = U @ np.diag(S) @ Vt
    return E
