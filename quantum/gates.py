# This module is dedicated to long implementations of quantum gates used in the DQI algorithm.

from classical.binary_tree import BinaryTree
from qiskit.circuit.library import RYGate
import numpy as np


def multi_controlled_gate(circuit, gate, target, controls, ancillas):
    """Apply a multi-controlled gate to a target qubit with specified controls and ancillas. 
    Handles single control aswell. 

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        gate (Gate): The gate to apply.
        target (Qubit): The target qubit for the gate.
        controls (list of Qubit): The control qubits.
        ancillas (QuantumRegister): Ancilla register for intermediate computations.

    Returns:
        None
    """
    k = len(controls)
    assert len(ancillas) >= max(0, k - 1), "Not enough ancillas for the number of controls."

    if k == 1:
        # Single control case
        circuit.append(gate.control(1), [controls[0], target])
    elif k > 1:
        # Multi-control case
        ancilla_flag = ancillas[k - 2]
        circuit.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, k-1):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])
        circuit.ccx(controls[k-1], ancillas[k-3], ancilla_flag)

        # control on ancilla_flag - the AND of all controls
        circuit.append(gate.control(1), [ancilla_flag, target])

        # uncompute ancillas
        circuit.ccx(controls[k-1], ancillas[k-3], ancilla_flag)
        for i in range(k-2, 0, -1):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])
        circuit.ccx(controls[0], controls[1], ancillas[0])
            


def state_prep_unitary(circuit, qubits, vector, ancilla_register):
    """This function implements a unitary that prepares the quantum state for the given vector from the statee |0>^n.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        qubits (list of Qubit or QuantumRegister): The qubits that we want to prepare the state on.
        vector (np.ndarray): The vector that gives the state, must be of length 2^n.
        ancilla_register (QuantumRegister): Ancilla register for intermediate computations.
    
    Returns:
        None
    """
    assert 2**len(qubits) == len(vector), "Vector length must be 2^{number of qubits}."
    tree = BinaryTree()
    tree.build_tree(vector)

    for i in range(len(qubits)):
        for node in tree.levels[i]:
            if hasattr(node, 'left') and hasattr(node, 'right') and node.value > 0:
                theta = 2*np.arccos(np.sqrt(node.left / node.value))
                
                if len(node.path) == 0:
                    # No controls
                    circuit.ry(theta, qubits[i])
                else:
                    control_qubits = qubits[:len(node.path)]
                    qubits_to_flip = [control_qubits[j] for j, bit in enumerate(node.path) if bit == 0]
                    print(control_qubits)
                    # Apply X gates for negative controls
                    for qubit in qubits_to_flip:
                        circuit.x(qubit)
                    
                    # Apply controlled gate
                    multi_controlled_gate(circuit, RYGate(theta), qubits[i], control_qubits, ancilla_register)
                    
                    # Undo X gates
                    for qubit in qubits_to_flip:
                        circuit.x(qubit)