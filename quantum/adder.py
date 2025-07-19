# This module is dedicated to transforming the state |y> |0> on the error and syndrome registers into the state |y> |B^T y>. 
# The currrent state of the two registers when this procedure is called is: 
# \sum_{k=0}^l w_k / sqrt(m choose k) \sum_{y \in field_p st |y| = k} \product_{i = 1, st y_i nonzero}^m fourier{g}_i(y_i) |y> |0^n>
# So we intend to add B^T y into the syndome register for each y in this superpositition. Clearly this means controlling on the error register.
# Each state |y> |B^T y> in this intedned superposition can be written as: |y^1, ..., y^m> |\sum_{i = 1}^m b_{i,1} y^i, ..., \sum_{i = 1}^m b_{i,n} y^i>
# where B = (b_{i,j}) is the matrix of values from F_p and y^i's are p-level quantum systems.
# We make the assumption in this implementation that p = 2^t for some t, so that we can use the binary representation of y^i's across ceil(log_2(p)) = log_2(p) qubits.

import numpy as np
from quantum.gates import QFT, inverse_QFT

def adder(circuit, error_register, syndrome_register, B, m, n, t):
    """
    Main function that adds B^T y to the syndrome register for each y on the error register.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        error_register (QuantumRegister or list): The quantum register containing the state |y>.
        syndrome_register (QuantumRegister or list): The quantum register to which we will add B^T y.
        B (numpy.ndarray): The matrix B whose transpose we will add to the syndrome register.
        m (int): The number of rows in B.
        n (int): The number of columns in B.
        t (int): ceil(log_2(p)) where our field elements are in F_p.
    
    Returns:
        None
    """
    assert len(error_register) == m * t, "Error register size does not match expected size based on m and t."
    assert len(syndrome_register) == n * t, "Syndrome register size does not match expected size based on n and t."
    for i in range(0, n):
        QFT(circuit, syndrome_register[i*t: (i+1)*t], 2**t, swap=False)
    for k in range(0,m):
        # apply a parrallel application of these additions
        for i in range(0, n):
            target = syndrome_register[i*t: (i+1)*t]  
            x = (i+k) % m
            control = error_register[x*t : (x+1)*t]
            b = B[x,i]
            if b != 0:
                controlled_add(circuit, target, control, b, t)
    for i in range(0, n):
        inverse_QFT(circuit, syndrome_register[i*t: (i+1)*t], 2**t, swap=False)

def controlled_add(circuit, target, control, b, t):
    """
    If the integer in the t control qubits is y then this function adds b*y to the t target qubits.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        target (list): The qubits to which we will add b*y.
        control (list): The qubits that control the addition.
        b (int): The matrix value.
        t (int): The number of qubits in the target and control registers.

    Returns:
        None
    """
    assert len(target) == len(control) == t
    # we apply the controlled rotations in parrallel to minimise depth
    for k in range(t):
        for j in range(k,t):
            targ = j - k
            ctrl = j
            angle = 2 * np.pi * b / (2 ** (k+ 1))
            circuit.cp(angle, control[ctrl], target[targ])