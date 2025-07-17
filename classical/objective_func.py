# This module contains the functions that calculate the objective functions of inputs.
# Namely we have f_i(), g_i(), fourier{g}_i and f(x_vec)

import numpy as np

def obj_f(index, input, Fs):
    """
    One component of the objective function to be maximised.
    The function f_i() which returns +1 if the input satisfies the ith constraint and -1 if it does not

    Args:
        input (int): an element of field_p
        index (int): the index of the constraint to check/the set F_i to check against
                     WARNING: the index to be inputted is based on F_1, ..., F_m not F_0, ..., F_{m-1}
        Fs (list): list of sets F_1, ..., F_m
    Returns:
        int: +1 if input satisfies the ith constraint, -1 otherwise
    """
    if input in Fs[index-1]:
        return 1
    else:
        return -1 

def obj_g(index, input, Fs, p, r):
    """
    One component of the equivalent objective function to obj_f. 
    Use this since fourier{g}_i(0) = 0 and sum_{y in field_p} |fourier{g}_i(y)|**2 = 1.
    
    Args:
        input (int): an element of field_p
        index (int): the index of the constraint to check/the set F_i to check against
                     WARNING: the index to be inputted is based on F_1, ..., F_m not F_0, ..., F_{m-1}
        Fs (list): list of sets F_1, ..., F_m
        p (int): the field size
        r (int): the size of each constraint set
    Returns:
        float: value in [-1,1]

    WARNING: f_bar and varphi hardcoded since we are working with a fixed r, would need to change this if no r
    """
    f_bar = (2*r / p) -1
    varphi = np.sqrt(4*r*(1 - r/p))

    return (obj_f(index, input, Fs) - f_bar)/ varphi

def fourier_g(index, input, Fs, p, r, field_p, omega):
    """
    One component of the fourier transform of the objective function g.

    Args:
        input (int): an element of field_p
        index (int): the index of the constraint of the corresponding component of g
        Fs (list): list of sets F_1, ..., F_m
        p (int): the field size
        r (int): the size of each constraint set
        field_p (set): the field elements
        omega (complex): primitive p-th root of unity
    Returns:
        float: the value of the fourier transform of g at input ie fourier{g}_i(input)
    """ 
    
     
    g_values = np.array([obj_g(index, x, Fs, p, r) for x in field_p]) 
    
    x_values = np.array(list(field_p))
    omega_powers = omega**(x_values * input) # vectorised operations
    
    # Element-wise multiplication and sum
    sum_result = np.sum(omega_powers * g_values)
    
    return sum_result / np.sqrt(p)

def objective_value(x_vec, Fs, m):
    """
    This function calcuates the objective value of an input vector - useful for checking a result of DQI is good.

    Args:
        x_vec (np.ndarray): a vector of values in F_p for which Bx_vec is hopefully in many of F_1, ... F_m
        Fs (list): list of sets F_1, ..., F_m
        m (int): number of constraints
    
    
    Returns:
        float: fraction of constraints satisfied by x_vec
    """
    # (note: obj_f uses 1-based indexing)
    results = [obj_f(i+1, x_vec[i], Fs) for i in range(len(x_vec))]
    return sum(results)/m