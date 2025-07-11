# This module carries out full diagonalisation of a matrix that we define that determines the weights of the intial state of the DQI algorithm.

import numpy as np
from utils.params import B, p, m, n, r
from classical.min_distance import get_l
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal


# the matrix is defined using parameters
def calc_weight_vec(l):
    d = (p - 2*r) / np.sqrt(r * (p - r))
    main_diag = [i * d for i in range(l+1)]
    a = [np.sqrt(k * (m - k + 1)) for k in range(1, l+1)]
    eigval, eigvec = eigh_tridiagonal(main_diag, a, select='i', select_range=(len(main_diag)-1, len(main_diag)-1)) # gets the largest eigenpair
    weight_vec = eigvec[:, 0]  / np.linalg.norm(eigvec[:, 0])
    weight_vec = np.concatenate((weight_vec, np.zeros(int(2**(np.ceil(np.log2(l+1)))) - len(weight_vec)))) # so it is described in ambient space
    return weight_vec

    
if __name__ == "__main__":
    l = get_l(B, p, m, n, r)
    weight_vec = calc_weight_vec(l)
    print("Weight vector:", weight_vec)