from typing import Optional, Union

import numpy as np
from qiskit.circuit.library import NLocal, ZZFeatureMap
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel


class KernelMethod:
    r"""
    Base class for kernel methods such as SVM and Ridge.
    Exploits quantum parallelization to efficiently compute
    kernel matrices for train and test data.
    """
    def __init__(self,
                 encoding_map: Optional[NLocal] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None):

        self.train_kernel_matrix = None
        self._encoding_map = encoding_map if encoding_map else ZZFeatureMap(2)
        self.quantum_instance = quantum_instance

    def _reset_train_matrix(self):
        """
        Resets stored train_kernel_matrix. Used in derived classes to control
        whether or not to recompute the training matrix
        """
        self.train_kernel_matrix = None

    def _compute_kernel_matrices(self, X_train, X_test):
        """
        Computes the kernel matrices of distances between each training datapoint
        and between training and test datapoints.
        Takes advantage of quantum circuits for faster computation.

        Args:
            X_train: the training data
            X_test: the unclassified input data
        Returns:
            ndarray of train and test kernel matrices
        """
        q_kernel = QuantumKernel(feature_map=self._encoding_map,
                                 quantum_instance=self.quantum_instance)

        if self.train_kernel_matrix is None:
            n_train = X_train.shape[0]

            # Train and test data stacked together to run backend only once
            X_total = np.vstack([X_train, X_test])

            total_kernel_matrix = q_kernel.evaluate(x_vec=X_train, y_vec=X_total)

            # Splitting the total matrix into training and test part
            self.train_kernel_matrix = total_kernel_matrix[:, 0:n_train]
            # Transposed for ease of use later on
            test_kernel_matrix = total_kernel_matrix[:, n_train:].T
        else:
            # Only the test kernel matrix is needed as the train one has already been computed
            test_kernel_matrix = q_kernel.evaluate(x_vec=X_test, y_vec=X_train)

        return self.train_kernel_matrix, test_kernel_matrix
