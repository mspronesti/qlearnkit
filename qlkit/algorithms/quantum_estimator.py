import logging
from typing import Union, Optional, List

import numpy as np
from abc import abstractmethod
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.result import Result
from sklearn.base import TransformerMixin
from qiskit.utils import QuantumInstance
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


class QuantumEstimator(TransformerMixin):
    def __init__(self,
                 encoding_map=None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None
                 ):
        """
        Args:
            encoding_map:
                Map to classical data to quantum states.
                This class does not impose any constraint on it. It
                can either be a custom encoding map or a qiskit FeatureMap
            quantum_instance:
                The quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`

        """
        self.X_train = np.asarray([])
        self.y_train = np.asarray([])
        self._encoding_map = encoding_map

        self._set_quantum_instance(quantum_instance)

    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray):
        """
        Fits the model using X as training dataset
        and y as training labels
        Args:
            X_train: training dataset
            y_train: training labels

        """
        raise NotImplementedError("Must have implemented this.")

    @abstractmethod
    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the labels associated to the
        unclassified data X_test
        Args:
            X_test: the unclassified data

        Returns:
            the labels associated to X_test
        """
        raise NotImplementedError("Must have implemented this.")

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the circuit."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self,
                         quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]]):
        """Quantum Instance setter"""
        self._set_quantum_instance(quantum_instance)

    def _set_quantum_instance(
            self,
            quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]]):
        """
        Internal method to set a quantum instance according to its type

        Args:
            The quantum instance to set. Can be a
            :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
            or a :class:`~qiskit.providers.BaseBackend`

        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance

    @property
    def encoding_map(self):
        """Returns the Encoding Map"""
        return self._encoding_map

    @encoding_map.setter
    def encoding_map(self, encoding_map):
        """Encoding Map setter"""
        self._encoding_map = encoding_map

    def execute(self,
                qcircuits: Union[QuantumCircuit, List[QuantumCircuit]]) -> Union[Optional[Result], None]:
        """
        Executes the given quantum circuit
        Args:
            qcircuits:
                a :class:`~qiskit.QuantumCircuit` or a list of
                this type to be executed

        Returns:
            the execution results
        """
        logger.info("Executing circuits...")
        if self._quantum_instance is None:
            raise QiskitError("Circuits execution requires a quantum instance")

        # Instead of transpiling and assembling the quantum object
        # and running the backend, we call execute from the quantum
        # instance that does it at once a very efficient way
        # please notice: this execution is parallelized
        result = self._quantum_instance.execute(qcircuits)
        return result

    @abstractmethod
    def score(self,
              X: np.ndarray,
              y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Returns a score of this model given samples and true values for the samples.
        In case of classification, this value should correspond to mean accuracy,
        in case of regression, the coefficient of determination :math:`R^2` of the prediction.
        In case of clustering, the `y` parameter is typically ignored.

        Args:
            X: array-like of shape (n_samples, n_features)

            y: array-like of labels of shape (n_samples,)

            sample_weight: array-like of shape (n_samples,), default=None
                The weights for each observation in X. If None, all observations
                are assigned equal weight.

        Returns:
            a float score of the model.
        """
        raise NotImplementedError("You should have implemented this.")
