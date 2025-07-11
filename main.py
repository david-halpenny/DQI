# This is the main file that runs the DQI algorithm components that are in other modules.

import numpy as np
from utils.params import B, p, m, n, r
from classical.min_distance import get_l, clear_cache
l = get_l(B, p, m, n, r)

import qiskit as qs
from qiskit.circuit.library import RYGate
from qiskit.circuit import ControlledGate
from utils.simulate import visualise, save_circuit_image, check_state

from classical.binary_tree import BinaryTree
from classical.calc_weights import calc_weight_vec
from quantum.gates import state_prep_unitary

#set up four registers of qubits - weight and mask are subregisters of error
total_qubits = int((m + n)*(np.ceil(np.log2(p))))
error_register = qs.QuantumRegister(int(m*(np.ceil(np.log2(p)))))
syndrome_register = qs.QuantumRegister(int(n*(np.ceil(np.log2(p)))))
weight_register = error_register[: int(np.ceil(np.log2(l+1)))]
mask_indices = [int(i * np.ceil(np.log2(p)) - 1) for i in range(1, m+1)] # least significant qubits on every np.ceil(np.log2(p)) qubits
mask_register = [error_register[j] for j in mask_indices]
ancilla_register = qs.QuantumRegister(len(weight_register) - 1)
qc = qs.QuantumCircuit(ancilla_register, error_register, syndrome_register)

weight_vec = calc_weight_vec(l)
state_prep_unitary(qc, weight_register, weight_vec, ancilla_register) # prepare the weight_vec state on the weight register, use ancilla for computations

visualise(qc) # for when I want to see the circuit is being built as we hope - for all parameter sizes
# check_state(qc) # for when I want to check the probabilistic state is what i would expect = for small parameter sizes
