# This module implements the unitary denoted by G that acts on the error register and is defined by:
# G= \otimes_{i=1}^{m} G_i ; G_i|0> = |0> , G_i|1> = \sum_{y \in field_p\{0} } \fourier_{g}_{i}(y) |y>
# Since we only care about how the unitary acts on these two comp basis states of each ceil(log2(p)) qubit subregister, we can use Householder reflections to implement it efficiently.

import numpy as np
from qiskit.circuit.library import PhaseGate, ZGate
from qiskit import QuantumCircuit, QuantumRegister
from classical.objective_func import fourier_g
from quantum.gates import state_prep_unitary, multi_controlled_gate
from utils.simulate import visualise, check_state

def unitary_G(circuit, error_register, m, p, r, field_p, omega, Fs, ancillas):
    """
    Applies the unitary G to the error register.
    
    Args: 
        circuit (QuantumCircuit): The quantum circuit that the error register is part of.
        error_register (QuantumRegister or list): The quantum register we are transforming the state of.
        m (int): The number of G_i unitaries/ the number of subregisters.
        p (int): The size of the field we are working over.
        r (int): The size of the sets in Fs.
        field_p (set or list): The field we are working over.
        omega (complex): The p^th root of unity.
        Fs (list): The list of sets F_i's for i =1, ..., m.
        ancillas (list): The qubits we will use as ancillas to perform computations on.
    Returns:
        None
    """
    subreg_size = int(np.ceil(np.log2(p)))
    assert len(error_register) == m * subreg_size, "Error register size does not match expected size based on m and p."
    assert len(ancillas) >= m * (subreg_size - 1), "Ancilla register size does not match that necessary for parallel application of G."
    for i in range(1, m+1):
        qubits = error_register[(i-1)*subreg_size:i*subreg_size]
        ancilla_qubits = ancillas[(i-1)*(subreg_size-1):i*(subreg_size-1)]
        sub_G(i, circuit, qubits, ancilla_qubits, p, r, field_p, omega, Fs)



def sub_G(i, circuit, qubits, ancilla_qubits, p, r, field_p, omega, Fs):
    """
    Applies the unitary G_i to the i^th subregister of the error register. Using the theory of Householder reflections, we can implement G_i efficiently.
    G_i can be split up into four gates via G_i = V_i Z_0 V_i^dagger D_i where we define these gates as follows:
    
    V_i : the state preparation unitary that prepares a state v_i from |0>^{ceil(log_2(p))}
    Z_0 : diag(-1, 1, ..., 1) \in M_{2^ceil(log_2(p))}(\mathbb{C})
    D_i : diag(1, e^{-i theta_i}, 1, ..., 1) for some angle theta_i 

    
    Args:
        i (int): The index of the subregister (1-based indexing).
        circuit (QuantumCircuit): The quantum circuit that the qubits are part of.
        qubits (list): The qubits of the i^th subregister.
        ancilla_qubits (list): The ancilla qubits for this subregister.
        p (int): The size of the field we are working over.
        r (int): The size of the sets in Fs.
        field_p (set or list): The field we are working over.
        omega (complex): The p^th root of unity.
        Fs (list): The vector of F_i's for i =1, ..., m.
    Returns:
        None
    """
    subreg_size  = int(np.ceil(np.log2(p)))
    assert len(qubits) == subreg_size, "Subregister size does not match expected size based on p."
    assert len(ancilla_qubits) >= subreg_size - 1, "Not enough ancillas for the subregister size."

    # STEP 0: Define v_i and e^{-i theta_i} = scalar
    scalar = np.conj(fourier_g(i, 1, Fs, p, r, field_p, omega))/ np.abs(np.conj(fourier_g(i, 1, Fs, p, r, field_p, omega)))
    N_i = np.sqrt(1 - fourier_g(i, 1, Fs, p, r, field_p, omega)*scalar - np.conj(fourier_g(i, 1, Fs, p, r, field_p, omega)*scalar) + np.sum([fourier_g(i, y, Fs, p, r, field_p, omega)*np.conj(fourier_g(i, y, Fs, p, r, field_p, omega)) for y in field_p if y != 0]))
    if np.isclose(N_i, 0, atol=1e-12):
        # then we are dealing with the zero vector and this means that the vector we are trying to reflect |1> into is itself - so we can skip entirely
        return
    v_i = np.zeros(2**int(np.ceil(np.log2(p))), dtype=complex)
    v_i[0] = 0
    v_i[1] = 1 - (fourier_g(i, 1, Fs, p, r, field_p, omega)*scalar)
    
    # Vectorized computation for indices 2 to p-1
    field_p_subset = [y for y in field_p if y >= 2]
    v_i[field_p_subset] = [-fourier_g(i, y, Fs, p, r, field_p, omega)*scalar for y in field_p_subset]
    v_i = v_i/N_i
    assert np.isclose(np.linalg.norm(v_i), 1.0, atol=1e-10), f"Norm is {np.linalg.norm(v_i)}, expected 1.0"

    # STEP 1: Implement D_i
    for j in range(subreg_size-1):
        circuit.x(qubits[j])
    multi_controlled_gate(circuit, PhaseGate(-np.angle(scalar)), qubits[subreg_size-1], qubits[:subreg_size-1], ancilla_qubits)
    for j in range(subreg_size-1):
        circuit.x(qubits[j])

    # STEP 2: Implement V_i^dagger
    # Create temporary registers for the state preparation circuit
    temp_qreg = QuantumRegister(len(qubits))
    if len(ancilla_qubits) > 0:
        temp_anc_reg = QuantumRegister(len(ancilla_qubits))
        temp_circuit = QuantumCircuit(temp_qreg, temp_anc_reg)
        temp_ancilla_list = list(temp_anc_reg)
    else:
        temp_circuit = QuantumCircuit(temp_qreg)
        temp_ancilla_list = []

    state_prep_unitary(temp_circuit, list(temp_qreg), v_i, temp_ancilla_list)
    circuit.compose(temp_circuit.inverse(), qubits + ancilla_qubits, inplace=True)

    # STEP 3: Implement Z_0
    for j in range(subreg_size):
        circuit.x(qubits[j])
    multi_controlled_gate(circuit, ZGate(), qubits[subreg_size-1], qubits[:subreg_size-1], ancilla_qubits)
    for j in range(subreg_size):
        circuit.x(qubits[j])

    # STEP 4: Implement V_i
    circuit.compose(temp_circuit, qubits + ancilla_qubits, inplace=True)

