import logging
from typing import Union, Optional, List, Callable

import numpy as np
from abc import abstractmethod
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.result import Result
from sklearn.base import ClassifierMixin, TransformerMixin
from qiskit.utils import QuantumInstance
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


class QuantumClassifier(ClassifierMixin, TransformerMixin):
    def __init__(self,
                 encoding_map=None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None
                 ):
        """
        Args:
            encoding_map:
                map to classical data to quantum states.
                This class does not impose any constraint on it. It
                can either be a custom encoding map or a qiskit FeatureMap
            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`

        """
        self.X_train = np.asarray([])
        self.y_train = np.asarray([])
        self.qcircuits = None
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

    @abstractmethod
    def _create_circuits(self,
                         X_train: np.ndarray,
                         X_test: np.ndarray) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """
        Creates the quantum circuit(s) to perform
        the classification process
        Args:
            X_test: input data to be classified

        Returns:
            quantum circuits
        """
        raise NotImplementedError("Must have implemented this.")

    @staticmethod
    def parallel_construct_circuits(construct_circuit_task: Callable,
                                    X_test: np.ndarray,
                                    task_args: list = None) -> List[QuantumCircuit]:
        """
        Wrapper helper to qiskit parallel_map used to parallely construct
        circuits if the algorithm allows it. See qiskit parallel_map
        for more

        Args:
            construct_circuit_task:
                the task constructing a single
                quantum circuit
            X_test:
                the test dataset
            task_args:
                the other (optional) parameters of the task

        Returns:
            The result list contains the value of
                ``construct_circuit_task(X_test, *task_args)`` for
                    each value in ``X_test``.

        """
        from qiskit.tools import parallel_map
        return parallel_map(construct_circuit_task,
                            X_test,
                            task_args=task_args)

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
            quantum_instance: the quantum instance to set. Can be a
                `QuantumInstance`, a `Backend` or a `BaseBackend`

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

    def execute(self, X_test) -> Union[Optional[Result], None]:
        """
        Executes the given quantum circuit
        Args:
            X_test: the unclassified input data

        Returns:
            the execution results
        """
        logger.info("Executing circuits...")
        if self._quantum_instance is None:
            raise QiskitError("Circuits execution requires a quantum instance")

        self.qcircuits = self._create_circuits(self.X_train, X_test)

        # Instead of transpiling and assembling the quantum object
        # and running the backend, we call execute from the quantum
        # instance that does it at once a very efficient way
        # please notice: this execution is parallelized
        result = self._quantum_instance.execute(self.qcircuits)
        return result
