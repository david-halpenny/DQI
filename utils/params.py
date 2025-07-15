# This module contains the parameters used throughout the DQI algorithm

import random
import numpy as np


np.random.seed(42)  # For reproducibility
# t= 1, m =20, n =19, seed =42 gives a l = 5 that we can see gates are working
# t= 1, m =10, n =9, seed =42 gives a l = 2 that we can measure
q, t = 2,1 # these determine p
p = q**t
m = int(10)
n = int(9)
assert m > n
field_p = range(p)  # Finite field F_p
omega = np.exp(2j * np.pi / p) # pth root of unity
Fs = [] 
r = p // 2  # fixed size of sets F_1, ... ,F_m (optional although the implementation of f_bar and varphi in obj_g are based on such an r existing)
for i in range(m):
    Fs.append(set(random.sample(field_p, r)))


B = np.random.randint(0, p, size=(m, n))
# having a row of zeroes would mean one of the constraints is useless - all variables would produce the same element of field_p as output
# ensure no zero rows in B
for row in range(m):
    if np.all(B[row, :] == 0):
        # replace zero row with random non-zero row
        B[row, :] = np.random.randint(1, p, size=n)


# having a cloumn of zeroes would mean one of the variables is useless - it has no effect on any of the constraints 
# ensure no zero columns in B
for col in range(n):
    if np.all(B[:, col] == 0):
        # Replace zero column with random non-zero column
        B[:, col] = np.random.randint(1, p, size=m)

