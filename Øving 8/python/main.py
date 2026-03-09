import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *
from estimate_E_ransac import *

K = np.loadtxt('../data/K.txt')
I1 = plt.imread('../data/image1.jpg')/255.0
I2 = plt.imread('../data/image2.jpg')/255.0
# matches = np.loadtxt('../data/matches.txt')
matches = np.loadtxt('../data/task4matches.txt') # Part 4

u1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
u2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

# Task 2: Estimate E
# E = ...
Kinv = np.linalg.inv(K)
B1 = Kinv @ u1
B2 = Kinv @ u2

E, inter_treshold = estimate_E_ransac(B1,B2,K,4,2000)
print(inter_treshold)

T1,T2,T3,T4=decompose_E(E)

eyep=np.array([[1,0,0,0],
      [0,1,0,0],
      [0,0,1,0]])

P1=K@eyep



# Task 3: Triangulate 3D points
candidates = [T1, T2, T3, T4]
best = None
best_count = -1

for T in candidates:
    P2 = K @ T[:3, :]
    X = triangulate_many(u1, u2, P1, P2)   


    X1 = X                                  
    X2 = T @ X                               

    ok = (X1[2, :] > 0) & (X2[2, :] > 0)
    count = np.sum(ok)

    if count > best_count:
        best_count = count
        best = T

T = best
P2 = K @ T[:3, :]
X = triangulate_many(u1, u2, P1, P2)


e=epipolar_distance(F_from_E(E, K),u1,u2)
plt.hist(e, bins=50)
plt.xlabel("Epipolar error (pixels)")
plt.ylabel("Count")
plt.title("Histogram of epipolar distances")



#
# Uncomment in Task 2
#
# np.random.seed(123) # Leave as commented out to get a random selection each time
# draw_correspondences(I1, I2, u1, u2, F_from_E(E, K), sample_size=8)

#
# Uncomment in Task 3
#
draw_point_cloud(X, I1, u1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])

plt.show()
