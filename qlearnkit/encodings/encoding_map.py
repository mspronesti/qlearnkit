import numbers
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit


class EncodingMap(ABC):
    """
    Abstract Base class for qlearnkit encoding maps
    """

    def __init__(self, n_features: int = 2) -> None:
        """
        Creates a generic Encoding Map for classical data
        of size `n_features`

        Args:
            n_features: number of features (default: 2)
        """
        if n_features <= 0:
            raise ValueError(f"Expected n_features > 0. Got {n_features}")
        elif not isinstance(n_features, numbers.Integral):
            raise TypeError(
                "n_features does not take %s value, enter integer value"
                % type(n_features)
            )

        self._num_features = n_features
        self._num_qubits = 0

    @abstractmethod
    def construct_circuit(self, x) -> QuantumCircuit:
        """construct and return quantum circuit encoding data"""
        raise NotImplementedError("Must have implemented this.")

    @property
    def num_qubits(self):
        """getter for number of qubits"""
        return self._num_qubits

    @property
    def num_features(self):
        """getter for number of features"""
        return self._num_features

    def _check_feature_vector(self, x):
        if len(x) != self.num_features:
            raise ValueError(f"Expected features dimension "
                             f"{self.num_features}, but {len(x)} was passed")
