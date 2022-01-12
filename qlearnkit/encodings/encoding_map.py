from abc import ABC, abstractmethod
import numpy as np


class EncodingMap(ABC):
    """
    Abstract Base class for qlearnkit encoding maps
    """
    def __init__(self):
        pass

    @abstractmethod
    def circuit(self, x):
        """return quantum circuit encoding data"""
        raise NotImplementedError("Must have implemented this.")

    @abstractmethod
    def n_qubits(self, x):
        """return number of required qubits for qauntum encoding"""
        raise NotImplementedError("Must have implemented this.")

    @abstractmethod
    def state_vector(self, x):
        """return state vector after quantum encoding"""
        raise NotImplementedError("Must have implemented this.")

    def encode_dataset(self, dataset):
        """
       This method encodes a dataset.
       Args:
           dataset (np.npdarray): A dataset. Rows represent data examples
       Returns:
          (np.ndarray): the encoded dataset
       """
        return np.array([self.state_vector(x) for x in dataset])

