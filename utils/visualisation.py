import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

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


