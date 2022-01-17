import logging
import warnings
from typing import Optional, Union
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from qiskit.utils import QuantumInstance
from qiskit.providers import BaseBackend, Backend
from qiskit.circuit.library import NLocal, ZZFeatureMap
from ..quantum_estimator import QuantumEstimator
from ..kernel_method_mixin import KernelMethodMixin

logger = logging.getLogger(__name__)


class QRidgeRegressor(KernelMethodMixin, RegressorMixin, QuantumEstimator):
    """
    The Quantum Kernel Ridge algorithm for regression.
    Maps datapoints to quantum states using a FeatureMap or similar
    QuantumCircuit.

    Example:

        ..  jupyter-execute::

            from sklearn.preprocessing import MinMaxScaler
            from qlearnkit.algorithms import QRidgeRegressor
            from qiskit import BasicAer
            from qiskit.utils import QuantumInstance, algorithm_globals
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            from qiskit.circuit.library import PauliFeatureMap

            seed = 42
            algorithm_globals.random_seed = seed

            quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                               shots=1024,
                                               optimization_level=1,
                                               seed_simulator=seed,
                                               seed_transpiler=seed)
            mms = MinMaxScaler()
            # Use iris data set for training and test data

            X, y = make_regression(n_features=3,
                                   n_samples=100,
                                   noise=1,
                                   random_state=seed)

            X = mms.fit_transform(X)

            encoding_map = PauliFeatureMap(3)

            qridge = QRidgeRegressor(
                gamma=1e-2,
                encoding_map=encoding_map,
                quantum_instance=quantum_instance
            )

            # use diabetes dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            qridge.fit(X_train, y_train)

            print(f"Testing accuracy: "
                  f"{qridge.score(X_test, y_test):0.2f}")

    """

    def __init__(self,
                 encoding_map: Optional[NLocal] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                 gamma: float = 1.0):
        """
        Creates a Quantum Ridge Regressor

        Args:
            encoding_map:
                map to classical data to quantum states.
                Default: :class:`~qiskit_machine_learning.circuit.library.ZZFeatureMap`

            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`

            gamma:
                regularization parameter (float, default 1.0)
        """
        encoding_map = encoding_map if encoding_map else ZZFeatureMap(2)
        super().__init__(encoding_map, quantum_instance)

        # Initial setting for _gamma
        # Numerical value is set in fit method
        self.gamma = gamma
        self.label_class_dict = None
        self.class_label_dict = None
        self.alpha = None
        self.bias = None
        self.n_classes = None

    def fit(self, X, y):
        """
        Fits the model using X as training dataset
        and y as training values. The actual computation
        is done at the predict stage to allow running
        the qiskit backend only once.

        Args:
            X: training dataset
            y: training labels

        """
        if np.any(X < 0) or np.any(X > 1):
            warnings.warn("Detected input values not in range 0<=X<=1:\n"
                          f"{X[X < 0]}\n"
                          f"{X[X > 1]}\n"
                          "QRidgeRegressor may perform poorly out of this range. "
                          "Rescaling of the input is advised.")

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        logger.info("setting training data: ")
        for _X, _y in zip(X, y):
            logger.info("%s: %s", _X, _y)
        # Sets the training matrix to None to signal it must be
        # recomputed again in case train data changes
        self._reset_train_matrix()

    def _compute_alpha(self, train_kernel_matrix):
        """
        Computes alpha parameters for data in the training set.
        Alpha parameters will be used as weights in prediction.
        Args:
            train_kernel_matrix:
                matrix of distances from each point to each point
                in the training set

        Returns:
            numpy ndarray of alpha parameters
        """
        n_train = train_kernel_matrix.shape[0]
        I = np.eye(n_train)
        K = train_kernel_matrix + self.gamma * I
        Y = self.y_train

        alpha = np.linalg.solve(K, Y)
        return alpha

    def _compute_predictions(self, train_kernel_matrix, test_kernel_matrix):
        """
        Uses kernel matrices to predict values. Based on Least Squares
        Ridge formulation.

        Args:
            train_kernel_matrix:
                matrix of distances between training datapoints
            test_kernel_matrix:
                matrix of distances between training and test datapoints

        Returns:
            numpy ndarray of predicted classes. Uses the internal representation
        """
        # Fit
        self.alpha = self._compute_alpha(train_kernel_matrix)
        # Predict
        K_test = test_kernel_matrix
        predictions = K_test @ self.alpha
        return predictions

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """
        Solves a Least Squares problem to predict value of input data.

        Args:
            X_test: the test data

        Returns:
            numpy ndarray of predicted labels
        """
        if self.X_train is None:
            raise NotFittedError(
                "This QRidgeRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")

        if np.any(X_test < 0) or np.any(X_test > 1):
            warnings.warn("Detected input values not in range 0<=X<=1:\n"
                          f"{X_test[X_test < 0]}\n"
                          f"{X_test[X_test > 1]}\n"
                          "QRidgeRegressor may perform poorly out of this range. "
                          "Rescaling of the input is advised.")

        logger.info("Computing kernel matrices...")
        train_kernel_matrix, test_kernel_matrix = self._compute_kernel_matrices(self.X_train, X_test)
        logger.info("Done.")

        logger.info("Computing predictions...")
        y_predict = self._compute_predictions(train_kernel_matrix, test_kernel_matrix)
        logger.info("Done.")
        return y_predict
