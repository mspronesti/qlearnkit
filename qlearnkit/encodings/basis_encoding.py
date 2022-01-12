from .encoding_map import EncodingMap
from qiskit import QuantumCircuit
import numpy as np


class BasisEncoding(EncodingMap):
    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Retrieves the quantum circuit encoding via
        Basis Encoding
        Args:
            x: the data vector to encode

        Returns:
            the quantum encoding circuit

        Note:
              All data values must be either 1s or 0s
        """
        if isinstance(x, list):
            x = np.array(x)

        if np.count_nonzero(x == 0) + np.count_nonzero(x == 1) != len(x):
            raise ValueError("All features must be either 0 or 1 for Basis Encoding")

        x = np.array(x)
        x_reversed = x[::-1]  # match Qiskit qubit ordering

        n_qubits = self.n_qubits(x)
        qc = QuantumCircuit(n_qubits)

        one_indices = np.where(x_reversed == 1)[0]
        for i in one_indices:
            qc.x(i)

        return qc

    def n_qubits(self, x):
        return len(x)

    def state_vector(self, x):
        return ''.join(str(s) for s in x)
