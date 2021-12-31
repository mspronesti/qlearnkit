from sklearn.exceptions import NotFittedError
from qiskit.providers import BaseBackend, Backend
from qiskit.result import Result
from qlkit.algorithms import QuantumClassifier
from qiskit.utils import QuantumInstance
from typing import Dict, List, Optional, Union
from qlkit.algorithms.qknn.qknn_circuit import *
import collections

from qlkit.encodings import EncodingMap

logger = logging.getLogger(__name__)


class QKNeighborsClassifier(QuantumClassifier):
    r"""
    The Quantum K-Nearest Neighbors algorithm for classification

    Note:
        The naming conventions follow the KNeighborsClassifier from
        sklearn.neighbors

    Example:

        Classify data using the Iris dataset.

        ..  jupyter-execute::

            import numpy as np
            from qlkit.algorithms import QKNeighborsClassifier
            from qlkit.encodings import AmplitudeEncoding
            from qiskit import BasicAer
            from qiskit.utils import QuantumInstance, algorithm_globals
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split

            seed = 42
            algorithm_globals.random_seed = seed

            quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                               shots=1024,
                                               optimization_level=1,
                                               seed_simulator=seed,
                                               seed_transpiler=seed)

            encoding_map = AmplitudeEncoding()

            # Use iris data set for training and test data
            X, y = load_iris(return_X_y=True)

            num_features = 2
            X = np.asarray([x[0:num_features] for x, y_ in zip(X, y) if y_ != 2])
            y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

            qknn = QKNeighborsClassifier(
                n_neighbors=3,
                quantum_instance=quantum_instance,
                encoding_map=encoding_map
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            qknn.fit(X_train, y_train)

            print(f"Testing accuracy: "
                  f"{qknn.score(X_test, y_test):0.2f}")

    """

    def __init__(self,
                 n_neighbors: int,
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
        super().__init__(encoding_map, quantum_instance)
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
        return np.sqrt(np.abs((counts_0 - counts_1) / self._quantum_instance.run_config.shots))

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

    def _majority_voting(self,
                         y_train: np.ndarray,
                         fidelities: np.ndarray) -> np.ndarray:
        r"""
        Performs the classical majority vote procedure of the
        K-Nearest Neighbors classifier with the :math:`k` nearest to
        determine class
        Args:
            y_train: the train labels
            fidelities: the list ``F`` of fidelities used as a measure of
                        distance

        Returns:
            a list of predicted labels for the test data
        Raises:
              ValueError if math:: \exists f \in F \ t.c. f \notin [0, 1]
        """
        if np.any(fidelities < -0.2) or np.any(fidelities > 1.2):
            raise ValueError("Detected fidelities values not in range 0<=F<=1:"
                             f"{fidelities[fidelities < -0.2]}"
                             f"{fidelities[fidelities > 1.2]}")

        predicted_labels = np.empty(
            shape=(len(fidelities)),
            dtype=int
        )

        for i, f in enumerate(fidelities):
            indices = sorted(range(len(f)), key=lambda k: f[k])
            indices.reverse()

            # extract the num_neighbors (i.e. k) significant to our voting
            voters = [y_train[i] for i in indices][:self.n_neighbors]
            majority_vote = collections.Counter(voters).most_common(1)[0][0]
            predicted_labels[i] = majority_vote

        return predicted_labels

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
            circuits_line = self.parallel_construct_circuits(
                construct_circuit,
                self.X_train,
                task_args=[xtest,
                           self._encoding_map,
                           "swap_test_qc_{}".format(i)]
            )
            circuits = circuits + circuits_line

        """circuits = [
            construct_circuit(xt, xtr)
            for xt in X_test
            for xtr in X_train
        ]"""

        logger.info("Done.")
        return circuits

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data."""
        if self.X_train is None:
            raise NotFittedError(
                "This QKNeighborsClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")
        #  construct
        #  |-> results
        #  |-> fidelities
        #  |-> vote
        circuits = self._construct_circuits(X_test)
        results = self.execute(circuits)
        fidelities = self._get_fidelities(results, len(X_test))

        return self._majority_voting(self.y_train, fidelities)
