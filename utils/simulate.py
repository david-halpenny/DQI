import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator

def save_circuit_image(circuit, filename, scale = 0.6):
    """Save the circuit as an image file.

    Args:
        circuit (QuantumCircuit): The quantum circuit to draw.
        filename (str): The output filename for the image.
        scale (float, optional): Scale factor for the image size.
    
    Returns:
        None
    """
    fig = circuit.draw('mpl', scale=scale, fold=False) # fold makes image wrap around
    fig.savefig(filename, dpi=300, bbox_inches='tight') # dpi controls resolution, bbox_inches='tight' removes extra whitespace
    plt.close(fig)

def visualise(circuit):
    """Quickly visualise a quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to visualise.
    
    Returns:
        None
    """
    print(circuit.draw('text'))


def check_state(circuit, register=None):
    """Measure circuit and display top results.
    If a register is provided, only measure that register.

    Args:
        circuit (QuantumCircuit): The quantum circuit to measure.
        register (QuantumRegister or list, optional): The register or list of qubits to measure.
    
    Returns:
        None
    """

    qc_measured = circuit.copy()
    
    if register is not None:
        creg = ClassicalRegister(len(register))
        qc_measured.add_register(creg)
        for idx, qubit in enumerate(register):
            qc_measured.measure(qubit, creg[idx])
        num_measured = len(register)
    else:
        qc_measured.measure_all()
        num_measured = circuit.num_qubits

    simulator = AerSimulator()
    result = simulator.run(qc_measured, shots=10000).result()
    counts = result.get_counts()

    print(f"Top measurement results ({num_measured} measured qubits, 10000 shots):")
    for outcome, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:50]:
        reversed_outcome = outcome[::-1]
        prob = count / 10000
        print(f"|{reversed_outcome}⟩: {count} times ({prob:.3f})")
    print(f"Total unique states observed: {len(counts)}")

