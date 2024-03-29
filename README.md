# Qlearnkit python library

[![Python Versions](https://img.shields.io/badge/Python-3.7&nbsp;|&nbsp;3.8&nbsp;|&nbsp;3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/mspronesti/qlearnkit)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/mspronesti/qlearnkit/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/mspronesti/qlearnkit/blob/master/.github/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/mspronesti/qlearnkit/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/qlearnkit/)
[![PypI Versions](https://img.shields.io/pypi/v/qlearnkit)](https://pypi.org/project/qlearnkit/#history)


Qlearnkit is a python library implementing some fundamental Machine Learning models and algorithms for a gated quantum computer, built on top of [Qiskit](https://github.com/Qiskit/qiskit)
and, optionally, [Pennylane](https://pennylane.ai/).

## Installation

We recommend installing `qlearnkit` with pip
```bash
pip install qlearnkit
```
**Note:** pip will install the latest stable qlearnkit.
However, the main branch of qlearnkit is in (not so active) development. If you want to test the latest scripts or functions please refer to [development notes](#development-notes).

### Optional Install
Via pip, you can install `qlearnkit` with the optional extension
packages dependent on `pennylane`. To do so, run
```bash
pip install qlearnkit['pennylane']
```

### Docker Image
You can also use qlearnkit via Docker building the image from the provided `Dockerfile`

```bash
docker build -t qlearnkit -f docker/Dockerfile .
```

then you can use it like this

```bash
docker run -it --rm -v $PWD:/tmp -w /tmp qlearnkit python ./script.py
```

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

train_size = 32
test_size = 8
n_features = 4  # all features

# Use iris data set for training and test data
X_train, X_test, y_train, y_test = load_iris(train_size, test_size, n_features)

quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                   shots=1024,
                                   optimization_level=1,
                                   seed_simulator=seed,
                                   seed_transpiler=seed)

encoding_map = AmplitudeEncoding(n_features=n_features)

qknn = QKNeighborsClassifier(
    n_neighbors=3,
    quantum_instance=quantum_instance,
    encoding_map=encoding_map
)

qknn.fit(X_train, y_train)

print(f"Testing accuracy: "
      f"{qknn.score(X_test, y_test):0.2f}")
```

## Documentation
The documentation is available [here](https://mspronesti.github.io/qlearnkit).

Alternatively, you can build and browse it locally as follows:

first make sure to have `pandoc` installed

```bash
sudo apt install pandoc
```

then run

```bash
make doc
```

then simply open `docs/_build/index.html` with your favourite browser, e.g.

```bash
brave docs/_build/index.html
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
make test
```

Make sure to run

```bash
pre-commit install
```

to set up the git hook scripts. Now `pre-commit` will run automatically on `git commit`!

## Acknowledgments
The Quantum LSTM model is adapted from this [article](https://towardsdatascience.com/a-quantum-enhanced-lstm-layer-38a8c135dbfa) from Riccardio Di Sipio, but the Quantum part
has been changed entirely according to the architecture described in this [paper](https://arxiv.org/pdf/2009.01783.pdf).

## License

The project is licensed under the [Apache License 2.0](https://github.com/mspronesti/qlearnkit/blob/master/LICENSE).
