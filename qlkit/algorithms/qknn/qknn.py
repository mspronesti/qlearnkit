import qiskit
from qiskit.providers import BaseBackend
from qiskit.result import Result
from qlkit.algorithms import QuantumClassifier

from typing import Dict, List
from qlkit.algorithms.qknn.qknn_circuit import *
import collections

logger = logging.getLogger(__name__)


class QKNeighborsClassifier(QuantumClassifier):
    """
    The Quantum K-Nearest Neighbors algorithm for classification

    Note:
        The naming conventions follow the KNeighborsClassifier from
        sklearn.neighbors

    Example:

        Classify data using the Iris dataset.

        ..  jupyter-execute::
            from qlkit.algorithms import QKNeighborsClassifier
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from qlkit.encodings import AmplitudeEncoding

            # preparing the parameters for the algorithm
            encoding_map = AmplitudeEncoding()
            backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')

            qknn = QKNeighborsClassifier(
                n_neighbors=3,
                encoding_map=encoding_map,
                backend=backend
            )

            X, y = load_iris(return_X_y=True)
            X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2 ])
            y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            qknn.fit(X_train, y_train)

            prediction = qknn.predict(X_test)
            print(prediction)
            print(y_test)

            print("Test Accuracy: {}".format(
                sum([1 if p == t else 0 for p, t in zip(prediction, y_test)]) / len(prediction)
            ))

    """

    def __init__(self,
                 n_neighbors: int,
                 encoding_map,
                 backend: BaseBackend,
                 shots: int = 1024,
                 optimization_level: int = 1):
        """
        encoding_map:
                        map to classical data to quantum states.
                        This class does not impose any constraint on it. It
                        can either be a custom encoding map or a qiskit FeatureMap
            backend:
                the qiskit backend to do the compilation & computation on
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
                If None or invalid value, level 1 will be chosen as default.
        """
        super().__init__(encoding_map, backend, shots, optimization_level)
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fits the model using X as training dataset
        and y as training labels
        Args:
            X: training dataset
            y: training labels

        """
        self.X_train = np.asarray(self.encoding_map.encode_dataset(X))
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
        return np.sqrt(np.abs((counts_0 - counts_1) / self.shots))

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
        all_counts = results.get_counts()

        fidelities = np.empty(
            shape=(test_size, train_size)
        )

        all_fidelities = list()
        for counts in all_counts:
            fidelity = self._compute_fidelity(counts)
            all_fidelities.append(fidelity)

        for i in range(test_size):
            shift = i * train_size
            fidelities[i] = all_fidelities[shift:train_size + shift]

        return fidelities

    def _majority_voting(self,
                         y_train: np.ndarray,
                         fidelities: np.ndarray) -> np.ndarray:
        """
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

    def _create_circuits(self,
                         X_train: np.ndarray,
                         X_test: np.ndarray) -> List[QuantumCircuit]:
        """
        Creates the circuits to be executed on
        the gated quantum computer for the classification
        process

        Args:
            X_train: the training data
            X_test: the unclassified input data

        """
        logger.info("Starting circuits construction ...")

        X_test = self.encoding_map.encode_dataset(X_test)
        circuits = [
            construct_circuit(xt, xtr)
            for xt in X_test
            for xtr in X_train
        ]

        logger.info("Done.")
        return circuits

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data."""
        # construct
        #  |-> results
        #  |-> fidelities
        #  |-> vote
        results = self.execute(X_test)
        fidelities = self._get_fidelities(results, len(X_test))
        return self._majority_voting(self.y_train, fidelities)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qlkit.encodings import AmplitudeEncoding

# preparing the parameters for the algorithm
encoding_map = AmplitudeEncoding()
backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')

qknn = QKNeighborsClassifier(
    n_neighbors=3,
    encoding_map=encoding_map,
    backend=backend
)

X, y = load_iris(return_X_y=True)
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
qknn.fit(X_train, y_train)

prediction = qknn.predict(X_test)
print(prediction)
print(y_test)

print("Test Accuracy: {}".format(
    sum([1 if p == t else 0 for p, t in zip(prediction, y_test)]) / len(prediction)
))
