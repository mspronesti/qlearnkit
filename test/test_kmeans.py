import numpy as np
import qiskit
from qlkit.algorithms import QKMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_qkmeans():
    backend = qiskit.BasicAer.get_backend('qasm_simulator')

    # initialising the qkmeans model
    qkmeans = QKMeans(
        n_clusters=3,
        backend=backend
    )
    X, y = load_iris(return_X_y=True)
    X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
    y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1)

    # Perform clustering
    qkmeans.fit(train_data, train_label)

    # Predict the nearest cluster of train data
    predictions = qkmeans.predict(train_data)

    # Check if clustering was performed well
    accuracy = sum([1 if p == t else 0 for p, t in zip(predictions, qkmeans.clusters)]) / len(predictions)
    assert accuracy > 0.99
