import numpy as np
import pytest

from qlearnkit.algorithms import QKNeighborsClassifier
from qlearnkit.encodings import AmplitudeEncoding

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals

seed = 42
algorithm_globals.random_seed = seed

sv_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator_statevector"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)

qasm_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator"),
    shots=100,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)

encoding_map = AmplitudeEncoding()

@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qknn_normalized(quantum_instance, quantum_instance_type):
    # initialising the qknn model
    qknn = QKNeighborsClassifier(
        n_neighbors=3,
        quantum_instance=quantum_instance,
        encoding_map=encoding_map
    )

    train_data = [
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
    ]
    train_labels = [
        1,
        1,
        -1,
        -1
    ]
    test_data = [
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
    ]

    qknn.fit(train_data, train_labels)
    qknn_prediction = qknn.predict(test_data)
    np.testing.assert_array_equal(qknn_prediction, [1, -1],
                                  f"Test failed with {quantum_instance_type}.\n"
                                  f"Expected [1, -1] but it was {qknn_prediction}")


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qknn_score(quantum_instance, quantum_instance_type):
    """Test case similar to qiskit machine learning
    testing approach"""

    # initialising the qknn model
    qknn = QKNeighborsClassifier(
        n_neighbors=3,
        quantum_instance=qasm_quantum_instance,
        encoding_map=encoding_map
    )

    num_inputs = 2
    # construct data
    num_samples = 5
    # pylint: disable=invalid-name
    X = algorithm_globals.random.random((num_samples, num_inputs))
    y = 1.0 * (np.sum(X, axis=1) <= 1)
    while len(np.unique(y)) == 1:
        X = algorithm_globals.random.random((num_samples, num_inputs))
        y = 1.0 * (np.sum(X, axis=1) <= 1)

    # fit to data
    qknn.fit(X, y)

    # score
    score = qknn.score(X, y)
    np.testing.assert_(score >= 0.8, f"Test failed with {quantum_instance_type}.\n"
                                     f"Expected score >= 80%, but it was {score}")


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qknn_str(
        quantum_instance,
        quantum_instance_type,
        n_samples=40,
        n_features=4,
        n_test_pts=10,
        n_neighbors=3,
        random_state=0
):
    """This test is adapted from scikit-learn
    test suite for `neighbors` package"""
    # Test k-neighbors classification
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)
    y_str = y.astype(str)

    knn = QKNeighborsClassifier(
        n_neighbors=n_neighbors,
        quantum_instance=quantum_instance,
        encoding_map=encoding_map
    )

    knn.fit(X, y)
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    np.testing.assert_array_equal(y_pred, y[:n_test_pts],
                                  f"Test failed with {quantum_instance_type}.\n"
                                  f"Expected {y[:n_test_pts]} but it was {y_pred}")

    # Test prediction with y_str
    knn.fit(X, y_str)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    np.testing.assert_array_equal(y_pred, y_str[:n_test_pts],
                                  f"Test failed with {quantum_instance_type}.\n"
                                  f"Expected {y_str[:n_test_pts]} but it was {y_pred}")
