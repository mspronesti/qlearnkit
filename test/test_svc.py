import pytest

import numpy as np
from qiskit.circuit.library import ZZFeatureMap

from qlearnkit.algorithms import QSVClassifier

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
    shots=1000,
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
def test_qsvc_binary(quantum_instance, quantum_instance_type):
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        quantum_instance=quantum_instance,
        encoding_map=encoding_map
    )

    train_data = [
        [12, 15, 0, 0],
        [7, 8, 0, 0],
        [0, 0, 12, 15],
        [0, 0, 7, 8],
    ]
    train_labels = [
        0,
        0,
        1,
        1
    ]
    test_data = [
        [10, 10, 0, 0],
        [0, 0, 10, 10],
    ]

    qsvc.fit(train_data, train_labels)
    qsvc_prediction = qsvc.predict(test_data)
    np.testing.assert_array_equal(qsvc_prediction, [0, 1])


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qsvc_multiclass(quantum_instance, quantum_instance_type):
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        quantum_instance=quantum_instance,
        encoding_map=encoding_map,
    )

    train_data = [
        [12, 15],
        [7, 0],
        [-5, -5],
    ]
    train_labels = [
        0,
        1,
        2
    ]
    test_data = [
        [16, 15],
        [6, 0],
        [-6, -6]
    ]

    qsvc.fit(train_data, train_labels)
    qsvc_prediction = qsvc.predict(test_data)
    np.testing.assert_array_equal(qsvc_prediction, [0, 1, 2])


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qsvc_text_labels(quantum_instance, quantum_instance_type):
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        quantum_instance=quantum_instance,
        encoding_map=encoding_map
    )

    train_data = [
        [12, 15],
        [7, 0],
        [-5, -5],
    ]
    train_labels = [
        'A',
        'B',
        'C'
    ]
    test_data = [
        [16, 15],
        [6, 0],
        [-6, -6]
    ]

    qsvc.fit(train_data, train_labels)
    qsvc_prediction = qsvc.predict(test_data)
    np.testing.assert_array_equal(qsvc_prediction, ['A', 'B', 'C'])


"""
The following tests are adapted from 
qiskit-machine-learning testing suite
"""
# globals to all the following tests
sample_train = np.asarray(
    [
        [3.07876080, 1.75929189],
        [6.03185789, 5.27787566],
        [6.22035345, 2.70176968],
        [0.18849556, 2.82743339],
    ]
)
label_train = np.asarray([0, 0, 1, 1])

sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
label_test = np.asarray([0, 1])

feature_map = ZZFeatureMap(feature_dimension=2, reps=2)


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_qsvc_float(quantum_instance, quantum_instance_type):
    """Test QSVC with float data"""
    # by default uses ZZ ad feature map
    qsvc = QSVClassifier(quantum_instance=quantum_instance)
    qsvc.fit(sample_train, label_train)
    score = qsvc.score(sample_test, label_test)

    np.testing.assert_equal(score, 0.5,
                            f"Test failed with {quantum_instance_type}.\n"
                            f"Expected score of 0.5, but it was {score}")


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_change_kernel(quantum_instance, quantum_instance_type):
    """Test QSVC adding parameters later"""
    qsvc = QSVClassifier()

    qsvc.encoding_map = feature_map
    qsvc.quantum_instance = quantum_instance
    qsvc.fit(sample_train, label_train)
    score = qsvc.score(sample_test, label_test)

    np.testing.assert_equal(score, 0.5,
                            f"Test failed with {quantum_instance_type}.\n"
                            f"Expected score of 0.5, but it was {score}")

# TODO: add tests changing `gamma`
