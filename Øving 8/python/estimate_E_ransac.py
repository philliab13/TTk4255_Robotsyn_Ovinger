import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *

def estimate_E_ransac(B1, B2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    #Step 1
    curr_E=None
    best_inlier_count=-1
    for i in range(num_trials):
        inlier_count=0
        #Step 1.1
        sample = np.random.choice(B1.shape[1], size=8, replace=False)
        E = estimate_E(B1[:,sample], B2[:,sample])

        #Step 1.2
        F=F_from_E(E,K)
        u1=K@B1
        u2=K@B2
        e=epipolar_distance(F,u1,u2)
        #Step 1.3
        inlier_count = np.sum(e < distance_threshold)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            curr_E = E
    
    return curr_E, best_inlier_count
        


    
