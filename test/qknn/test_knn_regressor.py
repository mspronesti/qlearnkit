import numpy as np
import pytest

from qlearnkit.algorithms import QKNeighborsRegressor
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


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qknn_regressor(
    quantum_instance,
    quantum_instance_type,
    n_samples=40,
    n_features=4,
    n_test_pts=10,
    n_neighbors=3,
    random_state=0
):
    # Test k-neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    encoding_map = AmplitudeEncoding()

    knn = QKNeighborsRegressor(
        n_neighbors=n_neighbors,
        quantum_instance=quantum_instance,
        encoding_map=encoding_map
    )

    knn.fit(X, y)
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert np.all(abs(y_pred - y_target) < 0.3)
