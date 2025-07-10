# This module implements an algorithm for calculating the minimum distance of the code defined by Parity Check Matrix B^T
# Attempts have been made to optimise this as much as possible, since it scales exponentially like not using Galois library and doing Gaussian elimination ourselves
# Still will be a bottleneck of algorithm

import numpy as np
from itertools import combinations
from utils.params import p, B, m, n, r
from joblib import Memory
import os

cache_dir = os.path.join(os.path.dirname(__file__), '.joblib_cache')
memory = Memory(cache_dir, verbose=0)

def gaussian_elimination_mod_p(matrix):
    """Perform Gaussian elimination modulo p until we know whether the matrix is full rank or not.
    
    Args:
        matrix (np.ndarray): Input matrix to perform Gaussian elimination on
    Returns:
        bool: True if the matrix is full rank, False otherwise
    """
    A = matrix.copy()
    m, n = A.shape
    rank = 0 # we use the rank as an index to keep track of the row we are trying to create a leading one in
    
    for col in range(min(m, n)):
        # Find first possible pivot
        pivot_row = None
        for row in range(rank, m):
            if A[row, col] % p != 0: # checking each entry of the column for non-zero value
                pivot_row = row
                break
        
        if pivot_row is None: # gone through all of the rows for this column after the current rank, we can't pick a row above the current rank since swapping such a row would mess up the column \ket{rank-1} we have just created to the left
        # all were zero, then matrix isn't full rank
            return False
            
        # Swap rows
        if pivot_row != rank:
            tmp = A[rank].copy()
            A[rank] = A[pivot_row]
            A[pivot_row] = tmp

        # Find multiplicative inverse first entry of pivot
        pivot_inv = pow(int(A[rank, col]), p-2, p)  # using Fermat's Little Theorem we calculate the inverse of the first element of pivot row
        A[rank] = (A[rank] * pivot_inv) % p # make first element be 1 - leading ones necesseary
        
        # Over the next few lines we are going to eliminate the entries in this column other than the pivot/rank entry
        # Step 1: mask of rows ≠ rank and column entry ≠ 0 -don't want to change entries we don't need to
        mask = np.arange(m) != rank
        mask &= (A[:, col] % p) != 0

        # Step 2: extract the rows and factors
        rows_to_update = np.where(mask)[0] # gets indices of rows where mask is True
        factors = A[rows_to_update, col]  # gets the values/factors of these entries
        updates = (factors[:, None] * A[rank]) % p  # multiplies the pivot row by the factors mod p, vectorised so we have an entire matrix to minus of same size as A

        # Step 3: update rows in-place
        A[rows_to_update] = (A[rows_to_update] - updates) % p

        # after this step we should have the column \ket{rank} as the (rank)^th column

        rank += 1
        if rank == min(m, n):
            return True


def is_all_combinations_full_rank(H, t):
    """
    Check each possible combination of t columns in H to see if they are linearly independent - by checking if their submatrix is full rank.
    
    Args:
        H (galois.GF): Parity-check matrix in Galois field F_p.
        t (int): Size of the column subsets to check for linear independence.
    Returns:
        bool: True if all combinations of t columns are linearly independent, False otherwise.
    """

    cols = H.shape[1]
     # for each combination of t columns check the corresponding submatrix is full rank
    for col_idxs in combinations(range(cols), t):
        submatrix = H[:, col_idxs]
        if not gaussian_elimination_mod_p(submatrix):
            return False  # found a dependent subset
    return True


def find_max_t(H):
    """
    Uses binary search to find the maximum t such that all column subsets of size t are linearly independent.
    
    Args:
        H (galois.GF): Parity-check matrix in Galois field F_p.
    Returns:
        int: Maximum t such that all combinations of t columns are linearly independent.
    """
    low = 1
    high = min(H.shape[0], H.shape[1])  # Can't be more than number of columns

    max_t = 0
    while low <= high:
        mid = (low + high) // 2
        if is_all_combinations_full_rank(H, mid): # we know d-1 is at least mid (it could be mid itself), just we want to find the max possible value for d-1 not just one that works
            max_t = mid # we set the current value to the max, if we can't find a bigger value this will be final
            low = mid + 1
        else: # we know d-1 must be less than mid - there is a combination that is not linearly independent so we only need to check smaller size combinations since this combination exists for all sizes bigger than mid
            high = mid - 1
    # explanation for as algorithm terminates:
    # when high = low +1 then we have current max_t = low -1, we check low, 
    # if low works then change max_t and then check if high would work (ie check low+1 + high  //2)
    # if low doesn't work then we make high = low -1 so low < = high will fail and max_t will stay as it was
    return max_t # if we get max_t =0 there is a bug


def min_distance(PCM):
    """
    Given a parity-check matrix, calculates the minimum distance of the code defined by it.

    Args:
        PCM (np.ndarray): Parity-check matrix in Galois field F_p.
    Returns:
        int: Minimum distance of the code
    """
    max_t = find_max_t(PCM)
    return max_t + 1  # d = t + 1


# we calculate the error correcting ability l of the code defined by B^T
# since this computation is expensive we put it inside a function, so not automatically ran when importing from this module
# we also cache it once it has been performed for a given B, to save running the computation every time we want to import l into a new module
# import value of l via: import get_l, on next line l = get_l()
def get_l():
    """Get value of l based on current B in this module."""

    B_key = tuple(B.flatten()[:min(20, B.size)])  # cache only works with tuples
    return _calculate_l(B_key)

@memory.cache
def _calculate_l(B_key): # the input to this function forms the cache key
    """Calculate l based on B- persistently cached version"""
    print(f"Computing l for new B")
    
    d = min_distance(B.T)
    if (d-1)/2 > m*(1-r/p):
        l = m*(1-r/p)
    else:
        l = (d-1) // 2
    return l

def clear_cache():
    """Clear all cached results"""
    memory.clear()
    print("Cache cleared")


if __name__ == "__main__":
    B_T = B.T  # Transpose to get parity-check matrix
    min_dist = min_distance(B_T)
    print(f"Minimum distance of the code defined by the parity-check matrix B^T is {min_dist}", "PCM is", B_T)