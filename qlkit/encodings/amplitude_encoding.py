from .encoding_map import EncodingMap
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
"""Encoding classical data to quantum state via amplitude encoding."""


class AmplitudeEncoding(EncodingMap):
    """
    Amplitude Encoding algorithm
    """
    def circuit(self, x):
        n_qubits = self.n_qubits(x)
        q = QuantumRegister(n_qubits)

        qc = QuantumCircuit(q)
        qc.initialize(self.state_vector(x), [q[i] for i in range(n_qubits)])
        return qc

    def n_qubits(self, x):
        r"""
        Retrieves the number of needed qubits
        for the amplitude encoding, which is nothing
        but :math:`\lceil log_2{x}\rceil`

        Args:
            x: the vector to be encoded

        Returns:
           the number of required qubits
        """
        nqubits = np.log2(len(x))
        return int(nqubits) if nqubits % 1 == 0 else int(nqubits) + 1

    def state_vector(self, x):
        r"""
        The encoding of a state via amplitude encoding operates
        as follows:
        given a quantum state :math:`\psi`, it processes the data
        such that

        .. math::\langle\psi|\psi\rangle = 1,

        Args:
            x: the classical data

        Returns:
            (np.array) encoded quantum state
        """
        if isinstance(x, list):
            x = np.asarray(x)

        norm = np.linalg.norm(x)
        return x / norm if norm != 0 else x

