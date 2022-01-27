from sklearn.exceptions import NotFittedError

import logging
import numpy as np

from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from typing import Optional, Union
from sklearn.base import RegressorMixin

from .qknn_base import QNeighborsBase
from ...encodings import EncodingMap

logger = logging.getLogger(__name__)


class QKNeighborsRegressor(RegressorMixin, QNeighborsBase):
    """
    The Quantum K-Nearest Neighbors algorithm for regression

    Note:
        The naming conventions follow the KNeighborsRegressor from
        sklearn.neighbors
    """
    def __init__(self,
                 n_neighbors: int = 3,
                 encoding_map: Optional[EncodingMap] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):
        """
        Creates a QKNeighborsClassifier Object

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
        super().__init__(n_neighbors, encoding_map, quantum_instance)

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data."""
        if self.X_train is None:
            raise NotFittedError(
                "This QKNeighborsRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")

        circuits = self._construct_circuits(X_test)
        results = self.execute(circuits)

        # the execution results are employed to compute
        # fidelities which are used for the average
        fidelities = self._get_fidelities(results, len(X_test))

        logger.info("Averaging ...")

        k_nearest = self._kneighbors(self.y_train, fidelities)

        n_queries, _ = self.X_train.shape
        if n_queries == 1:
            predicted_labels = np.mean(k_nearest)
        else:
            predicted_labels = np.mean(k_nearest, axis=1)

        logger.info("Done.")
        return predicted_labels
