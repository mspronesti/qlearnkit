from .encoding_map import EncodingMap

import numpy as np
from functools import reduce

from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate

"""Encoding classical data to quantum state via amplitude encoding."""


class AngleEncoding(EncodingMap):
    """
    Angle Encoding algorithm. Assumes data is feature-normalized.
    """

    def __init__(
        self, n_features: int = 2, rotation: str = "Y", scaling: float = np.pi / 2
    ):
        r"""
        Args:
            rotation: the direction
                      admitted values: X, Y, Z
            scaling: scaling factor for normalized input data.
                     The default scaling :math:`\pi/2` does
                     not induce a relative phase difference.
        """
        ROT = {"X": RXGate, "Y": RYGate, "Z": RZGate}
        if rotation not in ROT:
            raise ValueError("No such rotation direction {}".format(rotation))

        super().__init__(n_features)
        # angle encoding requires 1 qubit
        # for each feature
        self._num_qubits = n_features
        self.gate = ROT[rotation]
        self.scaling = scaling

    def construct_circuit(self, x) -> QuantumCircuit:
        """
        Args:
            x (np.array): The input data to encode
        Returns:
            The circuit that encodes x.
            Assumes data is feature-normalized.
            Assumes every element in x is in [0, 1]

        """

        self._check_feature_vector(x)

        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            circuit.append(self.gate(2 * self.scaling * x[i]), [i])

        return circuit

    def state_vector(self, x):
        """
        The encoding of a state via angle encoding, operating
        a rotation around the ``rotation`` axis.

        Args:
             x (np.array): The input data to encode

        Returns:
             np.array: The state vector representation of x after angle encoding

        """

        qubit_states = []
        for x_i in x:
            qubit_state = self.gate(2 * self.scaling * x_i).to_matrix()[:, 0]
            qubit_states.append(qubit_state)

        return reduce(lambda a, b: np.kron(a, b), qubit_states)
