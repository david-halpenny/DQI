import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

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


def check_state(circuit):
    """Measure circuit and display top results"""
    import qiskit as qs
    
    # Add measurements
    qc_measured = circuit.copy()
    qc_measured.measure_all()
    
    # Run measurements
    simulator = AerSimulator()
    result = simulator.run(qc_measured, shots=1000).result()
    counts = result.get_counts()
    
    # Display top 10 results
    print(f"Top measurement results ({circuit.num_qubits} qubits, 1000 shots):")
    for outcome, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        reversed_outcome = outcome[::-1]
        prob = count / 1000
        print(f"|{reversed_outcome}⟩: {count} times ({prob:.3f})")
    
    print(f"Total unique states observed: {len(counts)}")