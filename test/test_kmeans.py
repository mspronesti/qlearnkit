import numpy as np
from qlkit.algorithms import QKMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals

seed = 42
algorithm_globals.random_seed = seed

quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                   shots=1024,
                                   optimization_level=1,
                                   seed_simulator=seed,
                                   seed_transpiler=seed)

def test_qkmeans():
    # initialising the qkmeans model
    qkmeans = QKMeans(
        n_clusters=3,
        quantum_instance=quantum_instance
    )
    X, y = load_iris(return_X_y=True)
    X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
    y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=42)

    # Perform clustering
    qkmeans.fit(train_data, train_label)

    # Predict the nearest cluster of train data
    predictions = qkmeans.predict(train_data)

    # Check if clustering was performed well
    accuracy = sum([1 if p == t else 0 for p, t in zip(predictions, qkmeans.labels_)]) / len(predictions)
    assert accuracy > 0.98
