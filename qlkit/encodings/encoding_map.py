from abc import ABC, abstractmethod
import numpy as np


class EncodingMap(ABC):
    """
    Abstract Base class for qlkit encoding maps
    This class and its derivatives are base on the following paper
    by Israel F. Araujo, Daniel K. Park, Francesco Petruccione and
    Adenilton J. da Silva:    https://arxiv.org/pdf/2008.01511.pdf
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

