import numpy as np
import qiskit

from qlkit.algorithms import QKNeighborsClassifier
from qlkit.encodings import AmplitudeEncoding

def test_qknn():
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    encoding_map = AmplitudeEncoding()

    # initialising the qknn model
    qknn = QKNeighborsClassifier(
        n_neighbors=3,
        backend=backend,
        encoding_map=encoding_map
    )

    train_data = [
        [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]
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

