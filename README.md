# Qlkit python library

<p align="center">
 <img alt="CI build" src="https://github.com/mspronesti/qlkit/actions/workflows/build-and-test.yml/badge.svg"/> 
 <!-- <img alt="License"  src="https://img.shields.io/github/license/mspronesti/qlkit"/> -->
 <!-- <img alt="Release"  src ="https://img.shields.io/github/v/release/mspronesti/qlkit"/> -->
</p> 


Qlkit (pronounced cool-kit) is a simple python library implementing well-know supervised and unsupervised machine learning algorithms for a gated quantum computer, build with [Qiskit](https://github.com/Qiskit/qiskit)

## Install Qlkit
Install Qlkit running 
```bash
sudo python3 setup.py install
```
In the near future, we will make it installable via pip.

## Getting started with Qlkit
Now that Qlkit is installed, it's time to begin working with the Machine Learning module. 
Let's try an experiment using the KNN Classifier algorithm to train and test samples from a 
data set to see how accurately the test set can be classified.

```python
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
X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2])
y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
qknn.fit(X_train, y_train)

print("Test Accuracy: {}".format(
    qknn.score(X_test, y_test)
))
```

## Developement notes

After cloning this repository, create a virtual environment

```bash
python3 -m venv .venv
```

and activate it

```bash
source .venv/bin/activate 
```

now you can install the requirements

```bash
pip install -r requirements-dev.txt
```

now run the tests

```bash
python -m pytest
```
