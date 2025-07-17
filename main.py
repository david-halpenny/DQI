# This is the main file that runs the DQI algorithm components that are in other modules.

import numpy as np
from utils.params import B, p, m, n, r, field_p, omega, Fs
from classical.min_distance import get_l, clear_cache
l = get_l(B, p, m, n, r)

import qiskit as qs
from utils.simulate import visualise, save_circuit_image, check_state

from classical.calc_weights import calc_weight_vec
from quantum.gates import state_prep_unitary
from quantum.binary_to_unary import binary_to_unary, position_mask_reg
from quantum.unary_to_dicke import unitary_lm
from scipy.special import comb
from quantum.unitary_G import unitary_G
from classical.objective_func import fourier_g
from utils.verify import calc_desired


# STEP 0: set up
#set up four registers of qubits - weight and mask are subregisters of error
error_register = qs.QuantumRegister(int(m*(np.ceil(np.log2(p)))))
syndrome_register = qs.QuantumRegister(int(n*(np.ceil(np.log2(p)))))
weight_register = error_register[: int(np.ceil(np.log2(l+1)))]
mask_indices = [int(i * np.ceil(np.log2(p)) - 1) for i in range(1, m+1)] # least significant qubits on every np.ceil(np.log2(p)) qubits
mask_register = [error_register[j] for j in mask_indices]

# since we uncompute each time, we only need the max number of ancillas that get used in parallel
# in state_prep_unitary we require len(weight_register) - 1
# in unitary_ll we require one and np.ceil(m/l) of these unitaries are applied in parallel 
# in each WDB we require one - but there can't be more than np.ceil(m/l) WDBs at any depth
# for parrallel application of the m G_i's we need ceil(log2(p)) ancillas per G_i. for ancilla efficiency we use the syndrome register as ancillas at the moment
ancilla_no = max(len(weight_register) - 1, int(np.ceil(m/l)), int((m*(np.ceil(np.log2(p)) -1) - n*(np.ceil(np.log2(p))))))
ancilla_register = qs.QuantumRegister(ancilla_no) # 
total_qubits = int((m + n)*(np.ceil(np.log2(p)))) + len(weight_register) - 1
qc = qs.QuantumCircuit(error_register, syndrome_register, ancilla_register)


# STEP 1: prepare a weighted superposition on the weight register that is optimal for DQIs performance
# ----> sum_{k = 0}^l (w_k * |k>) 
weight_vec = calc_weight_vec(l)
state_prep_unitary(qc, weight_register, weight_vec, ancilla_register) # prepare the weight_vec state on the weight register, use ancilla for computations
# visualise(qc)

# STEP 2: convert each |k> to |1^k, 0^{len(mask_register) - k}> on the mask register
# ----> sum_{k = 0}^l (w_k * |1^k, 0^{len(mask_register) - k}>)
position_mask_reg(qc, int(np.ceil(np.log2(l+1))), p, error_register)
# visualise(qc)
binary_to_unary(qc, int(np.ceil(np.log2(l+1))), l, mask_register)
# visualise(qc)

# STEP 3: apply U_l^m which transforms each unary representation of Hamming weight k in the superposition to a Dicke state of weight k 
unitary_lm(qc, mask_register, m, l, ancilla_register) 
# visualise(qc)

# STEP 4: apply G to the error register
unitary_G(qc, error_register, m, p, r, field_p, omega, Fs, ancillas = list(ancilla_register) + list(syndrome_register))
# visualise(qc)

# check_state(qc, error_register)
# visualise(qc)
# save_circuit_image(qc, "G_circuit.png")
