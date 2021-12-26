import numpy as np
from qlkit.algorithms import QKNeighborsClassifier
from qlkit.encodings import AmplitudeEncoding

from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals

seed = 42
algorithm_globals.random_seed = seed

quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                   shots=1024,
                                   optimization_level=1,
                                   seed_simulator=seed,
                                   seed_transpiler=seed)


def test_qknn():
    encoding_map = AmplitudeEncoding()

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
    np.testing.assert_array_equal(qknn_prediction, [1, -1])
