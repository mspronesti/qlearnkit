import logging
from typing import Optional, Union
import numpy as np
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from qiskit.utils import QuantumInstance
from qiskit.providers import BaseBackend, Backend
from qiskit.circuit.library import NLocal, ZZFeatureMap
from ..quantum_estimator import QuantumEstimator

logger = logging.getLogger(__name__)


class QSVClassifier(ClassifierMixin, QuantumEstimator):
    r"""
    The Quantum Support Vector Machine algorithm for classification.
    Maps datapoints to quantum states using a FeatureMap or similar
    QuantumCircuit.

    Example:

        Classify data using the Iris dataset.

        ..  jupyter-execute::

            import numpy as np
            from qlearnkit.algorithms import QSVClassifier
            from qiskit import BasicAer
            from qiskit.utils import QuantumInstance, algorithm_globals
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from qiskit.circuit.library import ZZFeatureMap

            seed = 42
            algorithm_globals.random_seed = seed

            quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                               shots=1024,
                                               optimization_level=1,
                                               seed_simulator=seed,
                                               seed_transpiler=seed)

            # Use iris data set for training and test data
            X, y = load_iris(return_X_y=True)

            num_features = 2
            X = np.asarray([x[0:num_features] for x, y_ in zip(X, y) if y_ != 2])
            y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

            encoding_map = ZZFeatureMap(2)

            qsvc = QSVClassifier(
                encoding_map=encoding_map,
                quantum_instance=quantum_instance
            )

            # use iris dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            qsvc.fit(X_train, y_train)

            print(f"Testing accuracy: "
                  f"{qsvc.score(X_test, y_test):0.2f}")

    """
    def __init__(self,
                 encoding_map: Optional[NLocal] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                 gamma: Union[float, str] = 'scale'):
        """
        Creates a Quantum Support Vector Classifier

        Args:
            encoding_map:
                map to classical data to quantum states.
                Default: :class:`~qiskit_machine_learning.circuit.library.ZZFeatureMap`
            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`
            gamma:
                regularization parameter
        """
        encoding_map = encoding_map if encoding_map else ZZFeatureMap(2)
        super().__init__(encoding_map, quantum_instance)

        # Initial setting for gamma
        # Numerical value is set in fit method
        self.gamma_init = gamma
        self.label_class_dict = None
        self.class_label_dict = None
        self.alpha = None
        self.bias = None
        self.n_classes = None
        self.train_kernel_matrix = None


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
        self.label_class_dict = None
        self.class_label_dict = None
        self.alpha = None
        self.bias = None

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.n_classes = np.unique(y).size
        n_features = self.X_train.shape[1]

        if self.gamma_init == 'scale':
            self.gamma = 1 / (n_features * np.var(self.X_train))
        elif self.gamma_init == 'auto':
            self.gamma = 1 / n_features
        elif isinstance(self.gamma_init, str):
            raise ValueError("Invalid argument value %s",self.gamma_init)
        else:
            self.gamma = self.gamma_init

        self.label_class_dict, self.class_label_dict = QSVClassifier._create_label_class_dicts(self.y_train)

        if self.n_classes == 1:
            raise ValueError("All samples have the same label")
        if self.n_classes == 2:
            classes_matrix = np.array(np.vectorize(self.label_class_dict.get)(self.y_train))
        else:
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
        # Sets the training matrix to None to signal it must be computed again
        self.train_kernel_matrix = None


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

    def _compute_alpha(self,train_kernel_matrix):
        """
        Computes alpha parameters for data in the training set.
        Alpha parameters will be used as weights in prediction.
        Internally distinguishes between binary and multiclass case
        Args:
            train_kernel_matrix:
                matrix of distances from each point to each point
                in the training set

        Returns:
            numpy ndarray of alpha parameters
        """
        n_train = train_kernel_matrix.shape[0]
        omega = train_kernel_matrix
        gamma_inv = 1 / self.gamma
        ones = np.ones(n_train)
        eye = np.eye(n_train)

        A = np.vstack([
            np.block([np.zeros(1), ones.reshape([1, n_train])]),
            np.block([ones.reshape([n_train, 1]), omega + gamma_inv * eye])
        ])
        if self.n_classes == 2:
            B = np.vstack([np.zeros(1), self.classes_train.reshape(-1,1)])
        else:
            B = np.vstack([np.zeros(self.n_classes), self.classes_train])

        # Binary case: X is a vector containing alpha values.
        # Multiclass case: X is a (n_train+1,n_classes) matrix
        # containing alpha values for each of the n_classes linear systems.
        # This is equivalent to solving n_classes distinct binary problems.
        X = np.linalg.solve(A, B)
        bias = X[0, :]
        alpha = np.squeeze(X[1:, :])
        return alpha, bias

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

    def _compute_predictions_multiclass(self, train_kernel_matrix, test_kernel_matrix):
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
        self.alpha, self.bias = self._compute_alpha(train_kernel_matrix)
        # Predict
        prediction_classes = np.argmax(test_kernel_matrix @ self.alpha + self.bias, axis=1)
        return prediction_classes

    def _compute_predictions_binary(self, train_kernel_matrix, test_kernel_matrix):
        """
        Uses kernel matrices to find the dividing hyperplane.
        Based on Least Squares Support Vector Machine formulation.
        Specialized case which uses a np.sign call instead of
        computing multiple hyperplanes and using argmax

        Args:
            train_kernel_matrix: matrix of distances between training datapoints
            test_kernel_matrix: matrix of distances between training and test datapoints

        Returns:
            numpy ndarray of predicted classes. Uses the internal representation
        """
        # Fit
        self.alpha, self.bias = self._compute_alpha(train_kernel_matrix)
        # Predict
        prediction_classes = np.sign(test_kernel_matrix @ self.alpha + self.bias)
        prediction_classes = (prediction_classes + 1) / 2
        return prediction_classes


    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """
        Solves a Least Squares problem to predict value of input data.
        Internally distinguishes between binary and multiclass case.
        For the binary case solves an optimization problem to find a
        dividing hyperplane.
        For the multiclass case uses a one-to-rest approach and thus
        needs to run the algorithm n_classes different times.

        Args:
            X_test: the test data

        Returns:
            numpy ndarray of predicted labels
        """
        if self.X_train is None:
            raise NotFittedError(
                "This QSVClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")

        logger.info("Computing kernel matrices...")
        train_kernel_matrix, test_kernel_matrix = self._compute_kernel_matrices(self.X_train, X_test)
        logger.info("Done.")

        logger.info("Computing predictions...")
        if self.n_classes == 2:
            classes_predict = self._compute_predictions_binary(train_kernel_matrix, test_kernel_matrix)
        else:
            classes_predict = self._compute_predictions_multiclass(train_kernel_matrix, test_kernel_matrix)

        # Converts back from internal numerical classes used in SVM
        # to user provided labels.
        y_predict = np.vectorize(self.class_label_dict.get)(classes_predict)
        logger.info("Done.")
        return y_predict
