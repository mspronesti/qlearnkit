from .encoding_map import EncodingMap

from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate
import numpy as np


"""Encoding classical data to quantum state via amplitude encoding."""


class AngleEncoding(EncodingMap):
    """
    Angle Encoding algorithm. Assumes data is feature-normalized.
    """

    def __init__(self, rotation='Y', scaling=np.pi / 2):
        r"""
        Args:
            rotation: the direction
                      admitted values: X, Y, Z
            scaling: scaling factor for normalized input data.
                     The default scaling .. math:: \pi/2 does
                     not induce a relative phase difference.
        """

        ROT = {
            'X': RXGate,
            'Y': RYGate,
            'Z': RZGate
        }

        if rotation not in ROT:
            raise ValueError('No such rotation direction {}'.format(rotation))

        self.gate = ROT[rotation]
        self.scaling = scaling

    def n_qubits(self, x):
        """
        Args:
            x (np.array): The input data to encode
        Returns:
            Number of qubits needed to encode x
        """

        return len(x)

    def circuit(self, x):
        """
        Args:
            x (np.array): The input data to encode
        Returns:
            (qiskit.QuantumCircuit): The circuit that encodes x
                Assumes data is feature-normalized.
                Assumes every element in x is in [0, 1].
        """
        n_qubits = self.n_qubits(x)
        Sx = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            Sx.append(self.gate(2 * self.scaling * x[i]), [i])

        return Sx

    def state_vector(self, x):
        """
        Args
             x (np.array): The input data to encode
        Returns:
             np.array: The state vector representation of x after angle encoding
        """
        from functools import reduce

        qubit_states = []
        for x_i in x:
            qubit_state = self.gate(2 * self.scaling * x_i).to_matrix()[:, 0]
            qubit_states.append(qubit_state)

        return reduce(lambda a, b: np.kron(a, b), qubit_states)
