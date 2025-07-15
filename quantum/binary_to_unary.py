# This module implements the subroutine for manipulating qubits so that the binary representation of the state is converted to a unary representation.

import numpy as np
from quantum.gates import swap_gate, controlled_swap
from qiskit import QuantumCircuit, QuantumRegister
from utils.params import p

def position_mask_reg(circuit, binary_dim, p, error_reg):
    """
    In this intermediate step we swap the qubits on the weight register so that they are in the following order on the mask register:
    |k_1, k_2, ..., k_t> ----> |0, k_t, 0, k_{t-1}, 0,0,0, k_{t-2}, ... , 0 ..., 0, k_1>
    where k_i is on qubit 2^(binary_dim - i) (starting from 1) of the mask register.
    This means that k_i is on qubit 2^(binary_dim - i)*np.ceil(np.log2(p)) (starting from 1) of the error register.
    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        binary_dim (int): The number of qubits used for the binary representation.
        error_reg (QuantumRegister): The full register that we are distributing the qubits onto.

    Returns:    
        None
    """
    assert len(error_reg) >= (int(np.ceil(np.log2(p))))*(2**binary_dim), "Total qubits must be 2^(binary_dim) to give space for the unary representation."

    i = 0
    j = binary_dim - 1
    # IDEA: we want to move the qubit in position i to position ceil(log2(p))*(2**(binary_dim - i)) - 1 (using zero-based indexing) 
    # this is because the error register is blocks of ceil(log2(p)) qubits and we want to position k_i on the (2**(binary_dim - i))^th qubit of the mask reg
    # as we move the first qubits we might reach the point where the qubit we want to swap with aren't |0> but are another qubit that we will soon be trying to position
    # so we swap from this qubit and work our way back through the qubits, until we reach the same problem, so we start swapping from the first of the qubits we have left to position and so on
    while i <= j:
        while (int(np.ceil(np.log2(p))))*(2**(binary_dim - i)) -1 > j  and i <= j:
            swap_gate(circuit, error_reg[i], error_reg[(int(np.ceil(np.log2(p))))*(2**(binary_dim - i)) - 1])
            i += 1
    
        while (int(np.ceil(np.log2(p))))*(2**(binary_dim - j) - 1) < i  and i <= j:
            swap_gate(circuit, error_reg[j], error_reg[(int(np.ceil(np.log2(p))))*(2**(binary_dim - j)) - 1])
            j -= 1
        
        # PREVENTING INFINITE LOOP: if the outer loop is true but neither inner loop is this could happen
        # if i < j, this isn't possible since at least one of the remaining qubits to move to the correct position must satisfy one of the while loop conditions.
        # if no qubits left to position did this would imply all their intended positions were being filled...by themselves just in the wrong order
        # this is impossible since none of the qubits we are rearranging are intended to be placed consecutively - which is how they are placed when they haven't been positioned yet
        # there would be a gap of at least one |0> beteen them even when p = 2

        # although when i=j it is possible that there is a remaining qubit to move to the right position
        # and their position is being filled... by themself (hence they don't satisfy either while loop)
        # so in fact we dont need to move anything and we just increment i
        if i == j and ((int(np.ceil(np.log2(p))))*(2**(binary_dim - i)) - 1) == i:
            i +=1


def binary_to_unary(circuit, binary_dim, l, qubits):
    """
    Uses the state after position_mask_reg has been applied: |0, k_t, 0, k_{t-1}, 0,0,0, k_{t-2}, ... , 0 ..., 0, k_1>
    and converts it to the unary representation |1^k, 0^(2^t-k)>, where k = k_1 + 2*k_2 + ... + 2^(t-1)*k_t.

    Args:
        circuit (QuantumCircuit): The quantum circuit to which the unary conversion will be applied.
        qubits (list): List of qubits that we assume to be in the state |k_1, k_2, ... , k_t>.
    
    Returns:
        None
    """
    
    # STEP 1: |0, k_t, 0, k_{t-1}, 0,0,0, k_{t-2}, ... , 0 ..., 0, k_1> ----> |0^k 1 0^(2^t-k-1)>
    circuit.x(qubits[0])  
    circuit.cx(qubits[1], qubits[0]) 
    for j in range(1, binary_dim -1):
        for i in range(2**j -1):
            controlled_swap(circuit, qubits[2**(j+1) - 1], qubits[i], qubits[2**(j) + i])
        for i in range(2**j - 1):
            circuit.cx(qubits[2**(j) + i], qubits[2**(j+1) - 1])
        circuit.cx(qubits[2**(j+1) - 1], qubits[2**(j) - 1])
    
    # early stopping to considerably reduce the number of gates necessary
    if l == 2**(binary_dim) -1:
        # then there is no early stopping
        for i in range(2**(binary_dim -1) - 1):
            controlled_swap(circuit, qubits[2**(binary_dim) - 1], qubits[i], qubits[2**(binary_dim-1) + i])
        for i in range(2**(binary_dim -1) - 1):
            circuit.cx(qubits[2**(binary_dim-1) + i], qubits[2**(binary_dim) - 1])
        circuit.cx(qubits[2**(binary_dim) - 1], qubits[2**(binary_dim-1) - 1])
    # l is by definition <= 2**(binary_dim) - 1
    else:
        for i in range(l - 2**(binary_dim - 1) +1):
            controlled_swap(circuit, qubits[2**(binary_dim) - 1], qubits[i], qubits[2**(binary_dim-1) + i])
        for i in range(l - 2**(binary_dim - 1) +1):
            circuit.cx(qubits[2**(binary_dim-1) + i], qubits[2**(binary_dim) - 1])
        # don't need to clear the very last qubit since we are guranteed to have made an effective swap so already cleared


    # STEP 2: |0^k 1 0^(2^t-k-1)> ----> |1^k 0^(2^t-k)>
    for i in range(l, 1, -1): # we know the single 1 must be on first l+1 qubits
        circuit.cx(qubits[i], qubits[i-1])
    # all states other than |10^(2^t-k-1> which corresponds to |0^t> will be of form |0 1^k 0^(2^t-k -)>
    circuit.x(qubits[1])
    circuit.cx(qubits[1], qubits[0])
    circuit.x(qubits[1])

    #cyclic swap
    for i in range(l):
        swap_gate(circuit, qubits[i], qubits[i+1])



def make_binary_state_register(t, p):
    """
    Creates a QuantumRegister with np.ceil(np.log2(p))*2^t qubits and initializes the first t qubits to |1>.
    Returns the QuantumCircuit and the list of qubits.
    """
    n_qubits = (int(np.ceil(np.log2(p))))*(2 ** t)
    qreg = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qreg)
    for i in range(t):
        qc.x(qreg[i])  # Set qubit i to |1>
    return qc, list(qreg)

