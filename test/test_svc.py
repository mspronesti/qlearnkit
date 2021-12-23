import pytest

import numpy as np
import qiskit
from qiskit.circuit.library import ZZFeatureMap
from qlkit.algorithms import QSVClassifier

def test_qsvc_binary():
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        backend=backend,
        encoding_map=encoding_map,
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


def test_qsvc_multiclass():
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        backend=backend,
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
        [16,15],
        [6, 0],
        [-6,-6]
    ]

    qsvc.fit(train_data, train_labels)
    qsvc_prediction = qsvc.predict(test_data)
    np.testing.assert_array_equal(qsvc_prediction, [0, 1, 2])

def test_qsvc_text_labels():
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    encoding_map = ZZFeatureMap(2)

    # initialising the qsvc model
    qsvc = QSVClassifier(
        backend=backend,
        encoding_map=encoding_map,
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
        [16,15],
        [6, 0],
        [-6,-6]
    ]

    qsvc.fit(train_data, train_labels)
    qsvc_prediction = qsvc.predict(test_data)
    np.testing.assert_array_equal(qsvc_prediction, ['A', 'B', 'C'])