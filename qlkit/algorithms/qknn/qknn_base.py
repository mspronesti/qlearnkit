import numbers
from abc import ABC
from typing import List, Dict, Optional, Union

import numpy as np
import logging

from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, Backend
from qiskit.result import Result
from qiskit.tools import parallel_map
from qiskit.utils import QuantumInstance

from ..quantum_estimator import QuantumEstimator
from ...circuits import SwaptestCircuit
from ...encodings import EncodingMap

logger = logging.getLogger(__name__)


class QNeighborsBase(QuantumEstimator, ABC):
    def __init__(self,
                 n_neighbors: int = 3,
                 encoding_map: Optional[EncodingMap] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        """
        Base class for Nearest Neighbors algorithms

        Args:
            n_neighbors:
                number of neighbors participating in the
                majority vote
            encoding_map:
                map to classical data to quantum states.
                This class does not impose any constraint on it.
            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`

        """
        super().__init__(encoding_map, quantum_instance)

        if n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value"
                % type(n_neighbors)
            )

        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fits the model using X as training dataset
        and y as training labels
        Args:
            X: training dataset
            y: training labels

        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        logger.info("setting training data: ")
        for _X, _y in zip(X, y):
            logger.info("%s: %s", _X, _y)

    def _compute_fidelity(self, counts: Dict[str, int]):
        r"""
        Computes the fidelity, used as a measure of distance,
        from a dictionary of counts, which refers to the swap
        test circuit having a test datapoint and a train
        datapoint as inputs employing the following formula

        .. math::

            \sqrt{\abs{\frac{counts[0] - counts[1]}{n\_shots}}}

        Args:
            counts: the counts resulting after the simulation

        Returns:
            the computed fidelity
        """
        counts_0 = counts.get('0', 0)
        counts_1 = counts.get('1', 0)

        return np.sqrt(np.abs(counts_0 - counts_1)/self._quantum_instance.run_config.shots)

    def _get_fidelities(self,
                        results: Result,
                        test_size: int) -> np.ndarray:
        r"""
        Retrieves the list of all fidelities given the circuit
        results, computed via the :func:`calculate_fidelities` method
        Args:
            results: the simulation results
            test_size: the size of the test dataset

        Returns:
            numpy ndarray of all fidelities
        """
        train_size = self.X_train.shape[0]
        all_counts = results.get_counts()  # List[Dict(str, int)]

        fidelities = np.empty(
            shape=(test_size, train_size)
        )

        for i, (counts) in enumerate(all_counts):
            fidelity = self._compute_fidelity(counts)
            # the i-th subarray of the ndarray `fidelities` contains
            # the values that we will use for the majority voting to
            # predict the label of the i-th test input data
            fidelities[i // train_size][i % train_size] = fidelity

        return fidelities

    @staticmethod
    def _construct_circuit(feature_vector_1: np.ndarray,
                           feature_vector_2: np.ndarray,
                           encoding_map: EncodingMap = None) -> QuantumCircuit:
        r"""
        Constructs a swap test circuit employing a controlled
        swap. For instance

        .. parsed-literal::

                         ┌───┐       ┌───┐┌─┐
                q_0: ────┤ H ├─────■─┤ H ├┤M├
                     ┌───┴───┴───┐ │ └───┘└╥┘
                q_1: ┤ circuit-0 ├─X───────╫─
                     ├───────────┤ │       ║
                q_2: ┤ circuit-1 ├─X───────╫─
                     └───────────┘         ║
                c: 1/══════════════════════╩═
                                           0


        where feature_vector_1 = [1,0], feature_vector_2 = [0, 1]

        A swap test circuit allows to measure the fidelity between two quantum
        states, which can be interpreted as a distance measure of some sort.
        In other words, given two quanutm states :math:`|\alpha\rangle, \ |\beta\rangle`
        it measures how symmetric the state :math:`|\alpha\rangle \otimes |\beta\rangle` is

        Args:
            feature_vector_1:
                first feature vector
            feature_vector_2:
                second feature vector
            encoding_map:
                the mapping to quantum state to
                extract a :class:`~qiskit.QuantumCircuit`

        Returns:
            swap test circuit

        """
        if len(feature_vector_1) != len(feature_vector_2):
            raise ValueError("Input state vectors must have same length to"
                             "perform swap test. Lengths were:"
                             f"{len(feature_vector_1)}"
                             f"{len(feature_vector_2)}")

        if encoding_map is None:
            raise ValueError("encoding map must be specified to construct"
                             "swap test circuit")

        qc_1 = encoding_map.circuit(feature_vector_1)
        qc_2 = encoding_map.circuit(feature_vector_2)

        qc_swap = SwaptestCircuit(qc_1, qc_2)
        return qc_swap

    def _construct_circuits(self,
                            X_test: np.ndarray) -> List[QuantumCircuit]:
        """
        Creates the circuits to be executed on
        the gated quantum computer for the classification
        process

        Args:
            X_test: the unclassified input data

        """
        logger.info("Starting parallel circuits construction ...")

        circuits = []
        for i, xtest in enumerate(X_test):
            # computing distance of xtest with respect to
            # each point in X_train
            circuits_line = parallel_map(
                QNeighborsBase._construct_circuit,
                self.X_train,
                task_args=[xtest,
                           self._encoding_map]
            )
            circuits = circuits + circuits_line

        logger.info("Done.")
        return circuits

    def _kneighbors(self,
                    y_train: np.ndarray,
                    fidelities: np.ndarray,
                    *,
                    return_indices=False):
        """
        Retrieves the training labels associated to the :math:`k`
        nearest neighbors and (optionally) their indices

        Args:
            y_train:
                the training labels

            fidelities:
                the fidelities array

            return_indices:
                (bool) weather to return the indices or not

        Returns:
            neigh_labels: ndarray of shape (n_queries, n_neighbors)
                Array representing the labels of the :math:`k` nearest points

            neigh_indices: ndarray of shape (n_queries, n_neighbors)
                Array representing the indices of the :math:`k` nearest points,
                only present if return_indices=True.
        """
        if np.any(fidelities < -0.2) or np.any(fidelities > 1.2):
            raise ValueError("Detected fidelities values not in range 0<=F<=1:"
                             f"{fidelities[fidelities < -0.2]}"
                             f"{fidelities[fidelities > 1.2]}")
        # sklearn naming
        n_queries, _ = fidelities.shape

        # extracting indices of the k nearest neighbors
        # from the sorted fidelities
        neigh_indices = np.argsort(fidelities, axis=1)[:, -self.n_neighbors:]

        neigh_labels = y_train[neigh_indices]
        if return_indices:
            return neigh_labels, neigh_indices
        else:
            return neigh_labels
