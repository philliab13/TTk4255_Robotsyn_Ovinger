import numpy as np

def epipolar_distance(F, u1, u2, eps=1e-12):

    
    u1 = u1 / (u1[2:3, :] + eps)
    u2 = u2 / (u2[2:3, :] + eps)

    
    l2 = F @ u1     
    l1 = F.T @ u2    

   
    num2 = np.sum(u2 * l2, axis=0)  # (n,)
    num1 = np.sum(u1 * l1, axis=0)  # (n,)

    
    den2 = np.sqrt(l2[0, :]**2 + l2[1, :]**2) 
    den1 = np.sqrt(l1[0, :]**2 + l1[1, :]**2) 

    d2 = num2 / den2  
    d1 = num1 / den1  

    return np.abs(d2+d1)/2
