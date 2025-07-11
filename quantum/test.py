import qiskit as qs
from utils.simulate import save_circuit_image, visualise

def create_massive_circuit():
    """Create a 15-qubit circuit with many layers"""
    qc = qs.QuantumCircuit(15, 15)
    
    # Layer 1: Initialize with Hadamards
    for i in range(15):
        qc.h(i)
    
    # Layer 2: Create entanglement chains
    for i in range(14):
        qc.cx(i, i+1)
    
    # Layer 3: Rotation gates with different angles
    angles = [0.1*i for i in range(15)]
    for i in range(15):
        qc.rx(angles[i], i)
        qc.ry(angles[i]*2, i)
        qc.rz(angles[i]*3, i)
    
    # Layer 4: Complex controlled operations
    for i in range(0, 14, 2):
        qc.cz(i, i+1)
        if i+2 < 15:
            qc.cry(0.5, i, i+2)
    
    # Layer 5: Toffoli gates
    for i in range(0, 13, 3):
        qc.ccx(i, i+1, i+2)
    
    # Layer 6: SWAP operations
    for i in range(0, 14, 4):
        qc.swap(i, i+2)
    
    # Layer 7: More complex multi-qubit gates
    for i in range(5):
        qc.cx(i, i+5)
        qc.cx(i+5, i+10)
    
    # Layer 8: Phase gates
    for i in range(15):
        qc.p(0.25*i, i)
        qc.s(i)
        qc.t(i)
    
    # Layer 9: More entanglement
    for i in range(7):
        qc.cx(i, 14-i)
    
    # Layer 10: Final rotations
    for i in range(15):
        qc.u(0.1*i, 0.2*i, 0.3*i, i)
    
    # Measurements
    qc.measure_all()
    
    return qc

if __name__ == "__main__":
    massive = create_massive_circuit()
    
    visualise(massive) 
    save_circuit_image(massive, 'massive_circuit.png')