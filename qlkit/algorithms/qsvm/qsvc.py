import logging
from abc import ABC
from typing import Optional
import numpy as np
import sklearn.exceptions
from qiskit.utils import QuantumInstance
from qiskit.providers import BaseBackend
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import NLocal, ZZFeatureMap
from qlkit.algorithms import QuantumClassifier

logger = logging.getLogger(__name__)


class QSVClassifier(QuantumClassifier, ABC):
    r"""
    The Quantum Support Vector Machine algorithm for classification.
    Maps datapoints to quantum states using a FeatureMap or similar
    QuantumCircuit.

    Example:

        Classify data using the Iris dataset.

        ..  jupyter-execute::

            import qiskit
            import numpy as np
            from qlkit.algorithms import QSVClassifier
            from qiskit.providers import BaseBackend
            from qiskit.circuit.library import ZZFeatureMap
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            import matplotlib.pyplot as plt

            # preparing the parameters for the algorithm
            encoding_map = ZZFeatureMap(2)
            backend: BaseBackend = qiskit.Aer.get_backend('aer_simulator_statevector')

            qsvc = QSVClassifier(
                encoding_map=encoding_map,
                backend=backend
            )

            X, y = load_iris(return_X_y=True)
            X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
            y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            qsvc.fit(X_train, y_train)

            print("Test Accuracy: {}".format(
                qsvc.score(X_test, y_test)
            ))
    """

    def __init__(self,
                 encoding_map: Optional[NLocal],
                 backend: BaseBackend,
                 gamma: float = 1.,
                 shots: int = 1024,
                 optimization_level: int = 1,
                 seed: int = None):
        r"""
        Args:
            encoding_map:
                map to classical data to quantum states. Default: ZZFeatureMap
            backend:
                the qiskit backend to do the compilation & computation on
            gamma:
                regularization parameter
            shots:
                number of repetitions of each circuit, for sampling. Default: 1024
            optimization_level:
                level of optimization to perform on the circuits.
                Higher levels generate more optimized circuits,
                at the expense of longer transpilation time.
                        0: no optimization
                        1: light optimization
                        2: heavy optimization
                        3: even heavier optimization
                If None, level 1 will be chosen as default.
        """
        encoding_map = encoding_map if encoding_map else ZZFeatureMap(2)
        super().__init__(encoding_map, backend, shots, optimization_level)
        # TODO: include 'auto' and 'scale' options for gamma
        self.gamma = gamma
        self.train_kernel_matrix = None
        self.label_class_dict = None
        self.class_label_dict = None
        self.alpha = None
        self.bias = None
        self.n_classes = None
        self.seed = seed

    def fit(self, X, y):
        """
        Fits the model using X as training dataset
        and y as training labels. The actual computation
        is done at the predict stage to allow running
        the qiskit backend only once

        Args:
            X: training dataset
            y: training labels

        """
        self.train_kernel_matrix = None
        self.test_kernel_matrix = None
        self.label_class_dict = None
        self.class_label_dict = None
        self.alpha = None
        self.bias = None

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.n_classes = np.unique(y).size

        self.label_class_dict, self.class_label_dict = QSVClassifier._create_label_class_dicts(self.y_train)

        # Prepares an array of [+1,-1] values for each class
        # and organizes them in a matrix per the svm formulation.
        # This matrix notation will be useful later on to avoid nested for loops.
        classes_array = np.array([np.vectorize(self.label_class_dict.get)(self.y_train)])
        classes_array = classes_array.T
        classes_matrix = np.equal(classes_array,
                                  np.arange(self.n_classes) * np.ones((classes_array.size, self.n_classes)))

        self.classes_train = classes_matrix * 2 - 1
        logger.info("setting training data: ")
        for _X, _y in zip(X, y):
            logger.info("%s: %s", _X, _y)

    @staticmethod
    def _create_label_class_dicts(labels):
        """
        Creates dictionaries to convert from labels
        to classes used by svm. Classes are the integer values in range [0, 1, ..., n_classes]

        Args:
            labels: labels for which the dictionaries will be created

        Returns:
            dictionaries to convert from the user labels to the internal
            representation and vice versa
        """
        unique_labels = np.unique(labels)
        label_class_dict = {unique_labels[i]: i for i in range(unique_labels.size)}
        class_label_dict = {c: unique_labels[c] for c in range(unique_labels.size)}
        return label_class_dict, class_label_dict

    def _compute_kernel_matrices(self, X_train, X_test):
        """
        Computes the kernel matrices of distances between each training datapoint
        and between training and test datapoints.
        Takes advantage of quantum circuits for faster computation.

        Args:
            X_train: the training data
            X_test: the unclassified input data
        """
        n_train = self.X_train.shape[0]

        # TODO: add logic to handle already trained svm

        # Train and test data stacked together to run backend only once
        X_total = np.vstack([self.X_train, X_test])

        q_instance = QuantumInstance(self.backend,
                                     shots=self.shots,
                                     optimization_level=self.optimization_level,
                                     seed_simulator=self.seed,
                                     seed_transpiler=self.seed)
        q_kernel = QuantumKernel(feature_map=self.encoding_map,
                                 quantum_instance=q_instance)
        total_kernel_matrix = q_kernel.evaluate(x_vec=X_train, y_vec=X_total)

        # Splitting the total matrix into training and test part
        train_kernel_matrix = total_kernel_matrix[:, 0:n_train]
        # Transposed for ease of use later on
        test_kernel_matrix = total_kernel_matrix[:, n_train:].T

        return train_kernel_matrix, test_kernel_matrix

    # TODO: _compute_alpha method to divide _compute_predictions in two

    def _compute_predictions(self, train_kernel_matrix, test_kernel_matrix):
        """
        Uses kernel matrices to find n_classes dividing hyperplanes,
        following a one-to-rest approach. Based on Least Squares
        Support Vector Machine formulation. Actually solves n_classes
        linear systems in order to separate multiple classes.

        Args:
            train_kernel_matrix: matrix of distances between training datapoints
            test_kernel_matrix: matrix of distances between training and test datapoints

        Returns:
            numpy ndarray of predicted classes. Uses the internal representation
        """
        # Fit
        n_train = train_kernel_matrix.shape[0]
        omega = train_kernel_matrix
        gamma_inv = 1 / self.gamma
        ones = np.ones(n_train)
        eye = np.eye(n_train)

        A = np.vstack([
            np.block([np.zeros(1), ones.reshape([1, n_train])]),
            np.block([ones.reshape([n_train, 1]), omega + gamma_inv * eye])
        ])
        B = np.vstack([np.zeros(self.n_classes), self.classes_train])

        # X is a (n_train+1,n_classes) matrix containing alpha values
        # for each of the n_classes linear systems. This is equivalent
        # to solving n_classes distinct binary problems.
        X = np.linalg.solve(A, B)
        self.bias = X[0, :]
        self.alpha = X[1:, :]

        # Predict
        prediction_classes = np.argmax(test_kernel_matrix @ self.alpha + self.bias, axis=1)
        return prediction_classes

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """
        Solves a Least Squares problem to predict value of input data.
        Uses a one-to-rest approach and thus needs to run the algorithm
        n_classes different times

        Args:
            X_test: the test data

        Returns:
            numpy ndarray of predicted labels
        """
        if self.X_train is None:
            raise sklearn.exceptions.NotFittedError(
                "This QSVClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")

        logger.info("Computing kernel matrices...")
        self.train_kernel_matrix, self.test_kernel_matrix = self._compute_kernel_matrices(self.X_train, X_test)
        logger.info("Done.")

        logger.info("Computing predictions...")
        classes_predict = self._compute_predictions(self.train_kernel_matrix, self.test_kernel_matrix)

        # Converts back from internal numerical classes used in SVM
        # to user provided labels.
        y_predict = np.vectorize(self.class_label_dict.get)(classes_predict)
        logger.info("Done.")
        return y_predict
