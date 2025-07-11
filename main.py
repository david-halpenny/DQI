# This is the main file that runs the DQI algorithm components that are in other modules.

import numpy as np
from utils.params import B, p, m, n, r
from classical.min_distance import get_l, clear_cache
l = get_l(B, p, m, n, r)
import qiskit as qs

from utils.simulate import visualise, save_circuit_image, check_state

#set up four registers of qubits - weight and mask are subregisters of error
total_qubits = int((m + n)*(np.ceil(np.log2(p))))
error_register = qs.QuantumRegister(int(m*(np.ceil(np.log2(p)))))
syndrome_register = qs.QuantumRegister(int(n*(np.ceil(np.log2(p)))))
qc = qs.QuantumCircuit(error_register, syndrome_register)
weight_register = error_register[: int(np.ceil(np.log2(l+1))) +1]
mask_indices = [int(i * np.ceil(np.log2(p)) - 1) for i in range(1, m+1)] # least significant qubits on every np.ceil(np.log2(p)) qubits
mask_register = [error_register[j] for j in mask_indices]

