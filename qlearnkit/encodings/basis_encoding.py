from .encoding_map import EncodingMap
from qiskit import QuantumCircuit
import numpy as np


class BasisEncoding(EncodingMap):
    def __init__(self, n_features: int = 2):
        """
        Initializes Basis Encoding Map
        """
        super().__init__(n_features)
        # basis encoding requires 1 qubit
        # for each feature
        self._num_qubits = n_features

    def construct_circuit(self, x) -> QuantumCircuit:
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

        self._check_feature_vector(x)
        x = np.array(x)
        x_reversed = x[::-1]  # match Qiskit qubit ordering

        qc = QuantumCircuit(self.num_qubits)

        one_indices = np.where(x_reversed == 1)[0]
        for i in one_indices:
            qc.x(i)

        return qc

    def _check_feature_vector(self, x):
        if np.count_nonzero(x == 0) + np.count_nonzero(x == 1) != len(x):
            raise ValueError("All features must be either 0 or 1 for Basis Encoding.")
        super()._check_feature_vector(x)
