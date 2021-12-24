#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
import numpy as np
from qlkit.algorithms import QSVClassifier
from qiskit.providers import BaseBackend
from qiskit.circuit.library import ZZFeatureMap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# preparing the parameters for the algorithm
encoding_map = ZZFeatureMap(2)
backend: BaseBackend = qiskit.Aer.get_backend('aer_simulator_statevector')

qsvc = QSVClassifier(
    encoding_map=encoding_map,
    backend=backend
)

X, y = load_iris(return_X_y=True)
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
qsvc.fit(X_train, y_train)

print("Test Accuracy: {}".format(
    qsvc.score(X_test, y_test)
))


# In[2]:


import qiskit
import numpy as np
from qlkit.algorithms import QSVClassifier
from qiskit.providers import BaseBackend
from qiskit.circuit.library import ZZFeatureMap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# preparing the parameters for the algorithm
encoding_map = ZZFeatureMap(2)
backend: BaseBackend = qiskit.Aer.get_backend('aer_simulator_statevector')

qsvc = QSVClassifier(
    encoding_map=encoding_map,
    backend=backend
)

X, y = load_iris(return_X_y=True)
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
qsvc.fit(X_train, y_train)

print("Test Accuracy: {}".format(
    qsvc.score(X_test, y_test)
))

