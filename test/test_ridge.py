import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from qlearnkit.algorithms import QRidgeRegressor
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap

seed = 42
algorithm_globals.random_seed = seed

sv_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator_statevector"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
    optimization_level=1
)

qasm_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator"),
    shots=100,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
    optimization_level=1
)

def test_ridge_sv(
    quantum_instance=sv_quantum_instance,
    quantum_instance_type='statevector',
    n_samples=40,
    n_features=2,
    n_test_pts=10,
    random_state=0
):
    # Test ridge regression
    rng = np.random.RandomState(random_state)
    mms = MinMaxScaler()

    X, y = make_regression(n_features=n_features,
                           n_samples=n_samples,
                           noise=1,
                           random_state=seed)
    X = mms.fit_transform(X)

    y_target = y[:n_test_pts]

    encoding_map = PauliFeatureMap(n_features)

    ridge = QRidgeRegressor(
        gamma=1e-3,
        quantum_instance=quantum_instance,
        encoding_map=encoding_map,
    )

    ridge.fit(X, y)
    epsilon = 1e-6 * (2 * rng.rand(1, n_features) - 1)
    score = ridge.score(X[:n_test_pts] + epsilon,y_target)
    np.testing.assert_(score >= 0.8, f"Test failed with {quantum_instance_type}.\n"
                                     f"Expected score >= 80%, but it was {score}")


def test_ridge_qasm(
    quantum_instance=qasm_quantum_instance,
    quantum_instance_type='qasm',
    n_samples=40,
    n_features=2,
    n_test_pts=10,
    random_state=0
):
    # Test ridge regression
    rng = np.random.RandomState(random_state)
    mms = MinMaxScaler()

    X, y = make_regression(n_features=n_features,
                           n_samples=n_samples,
                           noise=1,
                           random_state=seed)
    X = mms.fit_transform(X)

    y_target = y[:n_test_pts]

    encoding_map = PauliFeatureMap(n_features)

    ridge = QRidgeRegressor(
        gamma=2.5,
        quantum_instance=quantum_instance,
        encoding_map=encoding_map,
    )

    ridge.fit(X, y)
    epsilon = 1e-6 * (2 * rng.rand(1, n_features) - 1)
    score = ridge.score(X[:n_test_pts] + epsilon,y_target)
    np.testing.assert_(score >= 0.8, f"Test failed with {quantum_instance_type}.\n"
                                     f"Expected score >= 80%, but it was {score}")


def test_change_kernel(
    quantum_instance=sv_quantum_instance,
    quantum_instance_type='statevector',
    n_samples=40,
    n_features=2,
    n_test_pts=10,
    random_state=0
):
    # Test ridge regression
    rng = np.random.RandomState(random_state)
    mms = MinMaxScaler()

    X, y = make_regression(n_features=n_features,
                           n_samples=n_samples,
                           noise=1,
                           random_state=seed)
    X = mms.fit_transform(X)

    y_target = y[:n_test_pts]

    encoding_map = PauliFeatureMap(n_features)

    ridge = QRidgeRegressor(
        gamma=1e-3,
    )
    ridge.quantum_instance = quantum_instance
    ridge.encoding_map = encoding_map

    ridge.fit(X, y)
    epsilon = 1e-6 * (2 * rng.rand(1, n_features) - 1)
    score = ridge.score(X[:n_test_pts] + epsilon,y_target)
    np.testing.assert_(score >= 0.8, f"Test failed with {quantum_instance_type}.\n"
                                     f"Expected score >= 80%, but it was {score}")

def test_change_gamma(
    quantum_instance=sv_quantum_instance,
    quantum_instance_type='statevector',
    n_samples=40,
    n_features=2,
    n_test_pts=10,
    random_state=0
):
    # Test ridge regression
    rng = np.random.RandomState(random_state)
    mms = MinMaxScaler()

    X, y = make_regression(n_features=n_features,
                           n_samples=n_samples,
                           noise=1,
                           random_state=seed)
    X = mms.fit_transform(X)

    y_target = y[:n_test_pts]

    encoding_map = PauliFeatureMap(n_features)

    ridge = QRidgeRegressor(
        quantum_instance=quantum_instance,
        encoding_map=encoding_map,
    )
    ridge.gamma = 10e-3

    ridge.fit(X, y)
    epsilon = 1e-6 * (2 * rng.rand(1, n_features) - 1)
    score = ridge.score(X[:n_test_pts] + epsilon,y_target)
    np.testing.assert_(score >= 0.8, f"Test failed with {quantum_instance_type}.\n"
                                     f"Expected score >= 80%, but it was {score}")
