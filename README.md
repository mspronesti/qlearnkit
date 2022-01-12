# Qlearnkit python library

[![Python Versions](https://img.shields.io/badge/Python-3.7&nbsp;|&nbsp;3.8&nbsp;|&nbsp;3.9-blue.svg?style=flat&logo=python&logoColor=white)]()
[![License](https://img.shields.io/github/license/mspronesti/qlearnkit)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/mspronesti/qlearnkit/actions/workflows/build-and-test.yml/badge.svg)]()
[![Upload Python Package](https://github.com/mspronesti/qlearnkit/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/qlearnkit)
[![PypI Versions](https://img.shields.io/pypi/v/qlearnkit)](https://pypi.org/project/qlearnkit/#history)

Qlearnkit is a simple python library implementing well-know supervised and unsupervised machine learning algorithms for a gated quantum computer, built with [Qiskit](https://github.com/Qiskit/qiskit).

## Installation

We recommend installing `qlearnkit` with pip
```bash
pip install qlearnkit
```
**Note:** pip will install the latest stable qlearnkit. 
However, the main branch of qlearnkit is in active development. If you want to test the latest scripts or functions please refer to [development notes](#development-notes).

## Getting started with Qlearnkit

Now that Qlearnkit is installed, it's time to begin working with the Machine Learning module. 
Let's try an experiment using the QKNN Classifier algorithm to train and test samples from a 
data set to see how accurately the test set can be classified.

```python
from qlearnkit.algorithms import QKNeighborsClassifier
from qlearnkit.encodings import AmplitudeEncoding
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals

from qlearnkit.datasets import load_iris

seed = 42
algorithm_globals.random_seed = seed

quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                   shots=1024,
                                   optimization_level=1,
                                   seed_simulator=seed,
                                   seed_transpiler=seed)

encoding_map = AmplitudeEncoding()

qknn = QKNeighborsClassifier(
    n_neighbors=3,
    quantum_instance=quantum_instance,
    encoding_map=encoding_map
)
# Use iris data set for training and test data

train_size = 32
test_size = 8
num_features = 4 # all features
 

X_train, X_test, y_train, y_test = load_iris(train_size, test_size, num_features)
qknn.fit(X_train, y_train)

print(f"Testing accuracy: "
      f"{qknn.score(X_test, y_test):0.2f}")
```

## Development notes

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
