from copy import deepcopy
from typing import List, Dict, Union, Optional

from qiskit.result import Result
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

from qlkit.algorithms.quantum_estimator import QuantumEstimator
from sklearn.base import ClusterMixin

from qlkit.algorithms.qkmeans.centroid_initialization import random, qkmeans_plus_plus, naive_sharding
from qlkit.algorithms.qkmeans.qkmeans_circuit import *
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class QKMeans(ClusterMixin, QuantumEstimator):
    """
    The Quantum K-Means algorithm for classification

    Note:
        The naming conventions follow the KMeans from
        sklearn.cluster

    Example:

        Classify data using the Iris dataset.

        ..  jupyter-execute::

            import numpy as np
            import matplotlib.pyplot as plt
            from qlkit.algorithms.qkmeans.qkmeans import QKMeans
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

            # Use iris data set for training and test data
            X, y = load_iris(return_X_y=True)

            num_features = 2
            X = np.asarray([x[0:num_features] for x, y_ in zip(X, y) if y_ != 2])
            y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

            qkmeans = QKMeans(n_clusters=3,
                              quantum_instance=quantum_instance
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            qkmeans.fit(X_train)

            # Plot the results
            colors = ['blue', 'orange', 'green']
            for i in range(X_train.shape[0]):
                plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[qkmeans.labels_[i]])
            plt.scatter(qkmeans.centroids[:, 0], qkmeans.centroids[:, 1], marker='*', c='g', s=150)
            plt.show()

            # Predict new points
            prediction = qkmeans.predict(X_test)
            print(prediction)

    """
    def __init__(self,
                 n_clusters: int = 6,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                 init: Union[str, np.ndarray] = "qkmeans++",
                 n_init: int = 1,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: int = 42
                 ):
        """
        Args:
            n_clusters:
                The number of clusters to form as well as the number of
                centroids to generate.
            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`
            init:
                Method of initialization of centroids.
            n_init:
                Number of time the qkmeans algorithm will be run with
                different centroid seeds.
            max_iter:
                Maximum number of iterations of the qkmeans algorithm for a
                single run.
            tol:
                Tolerance with regard to the difference of the cluster centroids
                of two consecutive iterations to declare convergence.
            random_state:
                Determines random number generation for centroid initialization.

        """
        super().__init__(quantum_instance=quantum_instance)
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.y_train = None
        self.X_train = None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        # do not rename : this name is needed for
        # `fit_transform` inherited method from
        # `ClusterMixin` base class
        self.labels_ = None

    def _init_centroid(self,
                       X: np.ndarray,
                       init: str,
                       random_state: int):
        """
        Initializes the centroids according to the following criteria:
        'qkmeans++: Create cluster centroids using the k-means++ algorithm.
        'random': Create random cluster centroids.
        'naive_sharding': Create cluster centroids using deterministic naive sharding algorithm.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        Args:
            X:
                Training dataset.
            init:
                Method of initialization of centroids.
            random_state:
                Determines random number generation for centroid initialization.
        """
        if init == "random":
            self.centroids = random(X, self.n_clusters, random_state)
        elif init == "qkmeans++":
            self.centroids = qkmeans_plus_plus(X, self.n_clusters, random_state)
        elif init == "naive":
            self.centroids = naive_sharding(X, self.n_clusters)
        else:
            self.centroids = init

    def _recompute_centroids(self):
        """
        Reassign centroid value to be the calculated mean value for each cluster.
        If a cluster is empty the corresponding centroid remains the same.
        """
        for i in range(self.n_clusters):
            if np.sum(self.labels_ == i) != 0:
                self.centroids[i] = np.mean(self.X_train[self.labels_ == i], axis=0)

    def _compute_distances_centroids(self,
                                     counts: Dict[str, int]) -> List[int]:
        """
        Compute distance, without explicitly measure it, of a point with respect
        to all the centroids using a dictionary of counts,
        which refers to the following circuit:

        .. parsed-literal::

                        ┌───┐                   ┌───┐
                |0anc>: ┤ H ├────────────■──────┤ H ├────────M
                        └───┘            |      └───┘
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘   |
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘

        Args:
            counts:
                Counts resulting after the simulation.

        Returns:
            The computed distance.
        """
        distance_centroids = [0] * self.n_clusters
        x = 1
        for i in range(0, self.n_clusters):
            binary = format(x, "b").zfill(self.n_clusters)
            distance_centroids[i] = counts[binary] if binary in counts else 0
            x = x << 1
        return distance_centroids

    def _get_distances_centroids(self,
                                 results: Result) -> np.ndarray:
        """
        Retrieves distances from counts via :func:`_compute_distances_centroids`

        Args:
            results: :class:`~qiskit.Result` object of execution results

        Returns:
            np.ndarray of distances
        """
        counts = results.get_counts()
        # compute distance from centroids using counts
        distances_list = list(map(lambda count: self._compute_distances_centroids(count), counts))

        return np.asarray(distances_list)

    def _construct_circuits(self,
                            X_test: np.ndarray) -> List[QuantumCircuit]:
        """
        Creates the circuits to be executed on
        the gated quantum computer for the classification
        process

        Args:
            X_test: The unclassified input data.

        Returns:
            List of quantum circuits created for the computation
        """
        logger.info("Starting circuits construction ...")
        '''
        circuits = []
        for xt in X_test:
            circuits.append(construct_circuit(xt, self.centroids, self.n_clusters))

        '''
        circuits = self.parallel_construct_circuits(
            construct_circuit,
            X_test,
            task_args=[self.centroids, self.n_clusters]
        )

        logger.info("Done.")
        return circuits

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None):
        """
        Fits the model using X as training dataset
        and y as training labels. For the qkmeans algorithm y is ignored.
        The fit model creates clusters from the training dataset given as input

        Args:
            X: training dataset
            y: Ignored.
               Kept here for API consistency

        Returns:
            trained QKMeans object
        """
        self.X_train = np.asarray(X)
        self._init_centroid(self.X_train, self.init, self.random_state)
        self.labels_ = np.zeros(self.X_train.shape[0])
        error = np.inf
        it = 0

        # while error not below tolerance, reiterate the
        # centroid computation for a maximum of `max_iter` times
        while error > self.tol and it < self.max_iter:
            # construct circuits using training data
            # notice: the construction uses the centroids
            # which are recomputed after every iteration
            circuits = self._construct_circuits(self.X_train)

            # executing and computing distances from centroids
            results = self.execute(circuits)
            distances = self._get_distances_centroids(results)

            # assigning clusters and recomputing centroids
            self.labels_ = np.argmin(distances, axis=1)
            centroids_old = deepcopy(self.centroids)
            self._recompute_centroids()

            # evaluating error and updating iteration count
            error = np.linalg.norm(self.centroids - centroids_old)
            it = it + 1

        return self

    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data.

        Args:
            X_test:
                New data to predict.

        Returns:
            Index of the cluster each sample belongs to.
        """
        if self.labels_ is None:
            raise NotFittedError(
                "This QKMeans instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator.")

        circuits = self._construct_circuits(X_test)
        results = self.execute(circuits)
        distances = self._get_distances_centroids(results)

        predicted_labels = np.argmin(distances, axis=1)
        return predicted_labels

    def score(self,
              X: np.ndarray,
              y: np.ndarray = None,
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Returns Mean Silhouette Coefficient for all samples.
        Args:
            X:  array of features

            y: Ignored.
               Not used, present here for API consistency by convention.

            sample_weight: Ignored.
                Not used, present here for API consistency by convention.

        Returns:
            Mean Silhouette Coefficient for all samples.
        """
        from sklearn.metrics import silhouette_score
        predicted_labels = self.predict(X)
        return silhouette_score(X, predicted_labels)
