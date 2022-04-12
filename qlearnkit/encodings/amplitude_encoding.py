import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

from .encoding_map import EncodingMap
from ..circuits.circuit_utils import to_basis_gates

"""Encoding classical data to quantum state via amplitude encoding."""


class AmplitudeEncoding(EncodingMap):
    """
    Amplitude Encoding map
    """

    def __init__(self, n_features: int = 2):
        super().__init__(n_features)
        n_qubits = np.log2(n_features)
        if not n_qubits.is_integer():
            # if number of qubits is not a positive
            # power of 2, an extra qubit is needed
            n_qubits = np.ceil(n_qubits)
        elif n_qubits == 0:
            # in this scenario, n_features = 1
            # then we need 1 qubit
            n_qubits = 1

        self._num_qubits = int(n_qubits)

    def construct_circuit(self, x) -> QuantumCircuit:
        """
        Constructs circuit for amplitude encoding

        Args:
            x: 1D classical data vector to be encoded
               must satisfy len(x) == num_features
        """
        self._check_feature_vector(x)
        if self.num_features % 2 != 0:
            # Number of features should be
            # a positive power of 2 for `initialize`,
            # then we add some padding
            x = np.pad(x, (0, (1 << self.num_qubits) - len(x)))

        state_vector = self.state_vector(x)

        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q)
        qc.initialize(state_vector, [q[i] for i in range(self.num_qubits)])

        # convert final circuit to basis gates
        # to unwrap the "initialize" block
        qc = to_basis_gates(qc)
        # remove the reset gates the unroller added
        qc.data = [d for d in qc.data if d[0].name != "reset"]
        return qc

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
        # retrieves the normalized vector
        # if the norm is not zero
        return x / norm if not norm == 0 else x
