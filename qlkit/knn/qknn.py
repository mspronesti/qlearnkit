import numpy as np

# required classes to conform to QuantumAlgorithm
# base class standard from qiskit
from typing import Dict, Optional, Union

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.providers import BaseBackend
from ._qknn_classifier import _QKNN_Classifier


class QKNN(QuantumAlgorithm):
    def __init__(self,
                 num_neighbors: 2,
                 training_dataset: None,
                 training_labels: None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]]) -> None:
        super().__init__(quantum_instance)

        self.num_neighbors = num_neighbors
        self.training_dataset = training_dataset
        self.training_labels = training_labels
        self.instance = _QKNN_Classifier(self)

    def _run(self) -> Dict:
        return self.instance.run()

    def fit(self, X, y):
        """
        Fits the model using X as training data
        and y as target labels
        Args:
            X: the training dataset
            y: the target labels
        """
        self.training_dataset = X
        self.training_labels = y

    def predict(self, data) -> np.ndarray:
        """
        Predict using the knn
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.

        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        return self.instance.predict(data)

    def majority_vote(self, labels, fidelities) -> np.ndarray:
        pass

    @property
    def ret(self) -> Dict:
        """
        Retrieves result(s)
        Returns:
            Dict: return value(s).
        """
        return self.instance.ret

    @ret.setter
    def ret(self, new_value):
        """
        Results setter
        Args:
            new_value: new value to set.
        """
        self.instance.ret = new_value
