#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
from qiskit.providers import BaseBackend
import numpy as np
from qlkit.algorithms import QKNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qlkit.encodings import AmplitudeEncoding

# preparing the parameters for the algorithm
encoding_map = AmplitudeEncoding()
backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')

qknn = QKNeighborsClassifier(
    n_neighbors=3,
    encoding_map=encoding_map,
    backend=backend
)

X, y = load_iris(return_X_y=True)
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2 ])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
qknn.fit(X_train, y_train)

prediction = qknn.predict(X_test)
print(prediction)
print(y_test)

print("Test Accuracy: {}".format(
    sum([1 if p == t else 0 for p, t in zip(prediction, y_test)]) / len(prediction)
))


# In[2]:


import qiskit
from qiskit.providers import BaseBackend
import numpy as np
from qlkit.algorithms import QKNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qlkit.encodings import AmplitudeEncoding

# preparing the parameters for the algorithm
encoding_map = AmplitudeEncoding()
backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')

qknn = QKNeighborsClassifier(
    n_neighbors=3,
    encoding_map=encoding_map,
    backend=backend
)

X, y = load_iris(return_X_y=True)
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2 ])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
qknn.fit(X_train, y_train)

prediction = qknn.predict(X_test)
print(prediction)
print(y_test)

print("Test Accuracy: {}".format(
    sum([1 if p == t else 0 for p, t in zip(prediction, y_test)]) / len(prediction)
))

