# This module implements a unitary U_l^m that transforms the unary representation |1^k0^{m-k}> to the Dicke state |D_k^m> for all k <= l.
# # There are multiple building blocks that we have to implement to achieve this.

from qiskit.circuit.library import RYGate
from quantum.gates import multi_controlled_gate, controlled_swap
import numpy as np
from scipy.special import comb
from typing import List, Dict


def unitary_ll(circuit, qubits, l, ancilla_qubit):
    """
    This function implements the unitary U_l^l a special case of the unitary U_l^m defined above.
    To build U_l^m we require m/l U_l^l 's (and other unitaries) so this is an important building block.
    We use the expression: U_l^l |x> = SCS(2) o SCS(3) o ... o SCS(l) |x> where SCS(k) is defined below. 
    
    Args:
        circuit (QuantumCircuit): The quantum circuit that the qubits we act on are a part of.
        qubits (list): The qubits that U_l^l acts on.
        l (int): The number of qubits that U_l^l acts on.
        ancilla_qubit (list of qubits): List of qubits that will act as ancillas - list has only one element.
    Returns:
        None
    """
    assert l == len(qubits)
    for k in range(l, 1, -1):
        scs(circuit, qubits, k, l, ancilla_qubit)  



def scs(circuit, qubits, k, l, ancilla_qubit):
    """
    This function implements a Split and Cyclic Shift operator, the building block for U_l^l.
    Each operator is parametrised by k will tells us the number of qubits it acts non-trivially on.

    Args:
        circuit (QuantumCircuit): The quantum circuit that the qubits we act on are a part of.
        qubits (list): The qubits that the operator acts on.
        k (int): The number of qubits that the operator acts non-trivially on.
        l (int): The total number of qubits that the operator acts on.
        ancilla_register (QuantumRegister): Ancilla register for intermediate computations.
    Returns:
        None
    """
    assert k >= 2
    circuit.cx(qubits[l-k+1], qubits[l-k])  
    alpha = 2*np.arccos(np.sqrt(1/k))
    circuit.cry(alpha, qubits[l-k], qubits[l-k+1])
    circuit.cx(qubits[l-k+1], qubits[l-k])

    if k > 2:    
        for i in range(l - k + 1, l-1):
            alpha = 2*np.arccos(np.sqrt((i - (l-k-1))/k))
            circuit.cx(qubits[i+1], qubits[l-k])
            multi_controlled_gate(circuit, RYGate(alpha), qubits[i+1], [qubits[l-k], qubits[i]], ancilla_qubit)
            circuit.cx(qubits[i+1], qubits[l-k])



def unitary_lm(circuit, mask_reg, m, l, ancilla_register):
    """
    This function implements the unitary U_l^m that transforms |1^k 0^{m-k}> to |D_k^m> for all k <= l.
    It is built of weight distribution blocks (WDBs) and unitaries U_l^l.
    Args:
        circuit (QuantumCircuit): The quantum circuit.
        mask_reg (QuantumRegister): The register whose state is to be transformed.
        m (int): The total number of qubits that the unitary acts on.
        l (int): The size of a block that each U_l^l will act on and create |D_k^l>.
        ancilla_register (QuantumRegister): Ancilla register for intermediate computations.
                                            Both the WDBs and U_l^l's need ancillas - one each.

    Returns:
        None
    """
    # assert m == len(mask_reg)
    x0 = m
    y0_multiple = int((x0/2)//l)
    y0 = l * y0_multiple if y0_multiple > 0 else l
    ancilla_dict = {} # this will keep track of how many ancillas we have assigned at each depth to allow for parallelisation
    
    log: List[Dict[str, object]] = []
    recursive_ordering(list(range(x0)), y0, l, 0, log) # this will update log with the details on how to apply the WDB blocks
    log.sort(key=lambda entry: entry["depth"]) 

    for entry in log:
        depth = entry["depth"]
        if depth not in ancilla_dict:
            ancilla_dict[depth] = 0
        ancilla_idx = ancilla_dict[depth]

        if len(entry['indices']) <= l:
            # then we can apply a U_l^l
            unitary_ll(circuit, [mask_reg[i] for i in entry['indices']], len(entry['indices']), [ancilla_register[ancilla_idx]])
        else:
            # we apply a WDB block
            x = len(entry['indices'])
            wdb(circuit, [mask_reg[i] for i in entry['indices']], x, entry['y'], l, [ancilla_register[ancilla_idx]])
        ancilla_dict[depth] += 1


def recursive_ordering(indices, y, min_qubits, depth, log):
    """
    This function recursively determines how the WDB blocks should be applied to our mask register so that we get blocks of size l and m mod l at the end.
    Each WDB block will be applied to x qubits and is parametrised by y - this function generates a list of these details to feed into unitary_lm.
    
    Args:
        indices (List[int]): The indices of the qubits that we are currently considering.
        y (int): The number of qubits that we will split off from the rest.
        min_qubits (int): We want all blocks of qubits in the deepest layer to be of this size (or the remainder of the total divided by this)
        depth (int): The current depth of the recursion.
        log (List[Dict[str, object]]): A list to record the qubit indices, the parameters y, and depth of the recursion.
        
    Returns:
        None
    """
    x = len(indices)

    # Record this unitary with depth
    log.append({"indices": indices.copy(), "y": y, "depth": depth})

    if x <= min_qubits:
        return  # base case: don't split further

    # Split into left (first x-y) and right (last y)
    left_size = x - y
    left_indices  = indices[:left_size]
    right_indices = indices[left_size:]

    # Determine new y values for each child
    left_multiple = int(((x - y) / 2) // min_qubits)
    left_y = min_qubits * left_multiple if left_multiple > 0 else int(x-y) % min_qubits
    # we have to ensure that if min_qubits <= x-y <= 2*min_qubits, that the y value for the next layer is x-y mod min_qubits so that the future x-y qubits is min_qubits in size

    right_multiple = int((len(right_indices) / 2) // min_qubits)
    right_y = min_qubits * right_multiple if right_multiple > 0 else int(len(right_indices)) % min_qubits

    # Recurse on each half, increasing depth
    recursive_ordering(left_indices,  left_y,  min_qubits, depth + 1, log)
    recursive_ordering(right_indices, right_y, min_qubits, depth + 1, log)


def wdb(circuit, qubits, x, y, l, ancilla_qubit, in_onehot = False):
    """
    This function implements a weight distribution block (WDB) applied to x qubits that transforms the state |1^k 0^{x-k}> 
    to the superposition of states: 
    1/sqrt(x choose k) sum_{i=0}^k sqrt((y choose i)*(x-y choose k-i))  |1^k-i  0^{x-y + i -k}> |1^i 0^{y-i}>
    for all k <= l.

    Clearly, by this definition we can see that the WDB splits off y qubits from the rest and distributes the weight k across both
    the first x-y qubits and the last y qubits in all possible unary representation ways.
    It does this in four steps.

    Args:
        circuit (QuantumCircuit): The quantum circuit that the qubits we act on are a part of.
        qubits (list): The qubits that the WDB acts on.
        x (int): The number of qubits that the WDB acts on.
        y (int): The number of qubits that will be split off from the rest.
        ancilla_qubit (list of qubits): The list of ancillas this WDB will use - the list will only have one element.
        in_onehot (bool): If True, the input state is in one-hot encoding.
    
    Returns:
        None
    """
    # we note that although the WDB is defined over x qubits, it only acts non-trivially on 2l of these: the first l of the x-y qubits and the first l of the y qubits.
    # considering l we have the initial state is |1^k 0^{l-k}> |0^{x-y-l}> |0^{l}> |0^{y-l}>
    # and from here on we will only write the 2l qubits we care about , so this state is |1^k 0^{l-k}> |0^{l}>.
    assert x-y >= l, "x-y must be at least l"
    if y < l:
        used_qubits = qubits[:l] + qubits[x-y:]
    else:
        used_qubits = qubits[:l] + qubits[x-y:x-y+l]

    # STEP 1: ----> |0^k-1 1 0^{l-k}> |0^{l}>
    # apply CNOT ladder
    if in_onehot == False: # might try to optimise for gates later since we were close to this state in the last step of the algorithm and brought it into unary
        for i in range(l-1):
            circuit.cx(used_qubits[i+1], used_qubits[i])

    # STEP 2: ----> 1/sqrt(x choose k) sum_{i=0}^k sqrt((y choose i)*(x-y choose k-i))  |0^k-1 1 0^{l-k}> |1^i 0^{l-i}>
    X = make_binom_matrix(x, y, l)
    for k in range(1,l+1):
        for j in range(1,k+1):
            # alpha_jk is computed as:
            # 2 * arccos(
            #     sqrt(
            #         binom(y, j-1) * binom(x-y, k-(j-1))
            #         -----------------------------------------
            #         sum_{i=j-1}^{k} binom(y, i) * binom(x-y, k-i)
            #     )
            # )
            alpha_jk = 2 * np.arccos(np.sqrt(X[j-1, k] / np.sum(X[j-1:k+1, k])))
            if j == 1:
                circuit.cry(alpha_jk, used_qubits[k-1], used_qubits[l])
            else:
                if j <= y: # make sure we don't try to apply a gate to a qubit that doesn't exist
                    multi_controlled_gate(circuit, RYGate(alpha_jk), used_qubits[l+j-1], [used_qubits[k-1], used_qubits[l+j-2]], ancilla_qubit)

    # STEP 3: ----> 1/sqrt(x choose k) sum_{i=0}^k sqrt((y choose i)*(x-y choose k-i))  |1^k 0^{l-k}> |1^i 0^{l-i}>
    # another CNOT ladder
    for i in range(l-2, -1, -1):
        circuit.cx(used_qubits[i+1], used_qubits[i])
    
    # STEP 4: ----> 1/sqrt(x choose k) sum_{i=0}^k sqrt((y choose i)*(x-y choose k-i))  |1^k-i 0^{l+i-k}> |1^i 0^{l-i}>
    for j in range(l):
        if l - j <= y:
            for i in range(j, 0, -1):
                controlled_swap(circuit, used_qubits[2*l -j - 1], used_qubits[l -i - 1], used_qubits[l-i])
            circuit.cx(used_qubits[2*l -j - 1], used_qubits[l-1])
        


def make_binom_matrix(x, y, l):
    i = np.arange(l+1).reshape(-1, 1)  # shape (l+1, 1)
    k = np.arange(l+1).reshape(1, -1)  # shape (1, l+1)
    binom_vec = np.frompyfunc(lambda n, r: comb(n, r, exact=True), 2, 1)
    X = binom_vec(y, i) * binom_vec(x - y, k - i)
    X = X.astype(int)
    valid = (k - i >= 0) & (k - i <= x - y) & (i <= y)
    X[~valid] = 0
    return X
