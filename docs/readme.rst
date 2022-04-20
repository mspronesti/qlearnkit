Qlearnkit python library
========================

|Python Versions| |License| |Build| |Upload Python Package| |PypI
Versions|

Qlearnkit is a simple python library implementing some fundamental
Machine Learning models and algorithms for a gated quantum computer,
built on top of `Qiskit <https://github.com/Qiskit/qiskit>`__ and,
optionally, `Pennylane <https://pennylane.ai/>`__.

Installation
------------

We recommend installing ``qlearnkit`` with pip

.. code:: bash

   pip install qlearnkit

**Note:** pip will install the latest stable qlearnkit. However, the
main branch of qlearnkit is in active development. If you want to test
the latest scripts or functions please refer to `development
notes <#development-notes>`__.

Optional Install
~~~~~~~~~~~~~~~~

Via pip, you can install ``qlearnkit`` with the optional extension
packages dependent on ``pennylane``. To do so, run

.. code:: bash

   pip install qlearnkit['pennylane']

Docker Image
~~~~~~~~~~~~

You can also use qlearnkit via Docker building the image from the
provided ``Dockerfile``

.. code:: bash

   docker build -t qlearnkit -f docker/Dockerfile .

then you can use it like this

.. code:: bash

   docker run -it --rm -v $PWD:/tmp -w /tmp qlearnkit python ./script.py

Getting started with Qlearnkit
------------------------------

Now that Qlearnkit is installed, it’s time to begin working with the
Machine Learning module. Let’s try an experiment using the QKNN
Classifier algorithm to train and test samples from a data set to see
how accurately the test set can be classified.

.. code:: python

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

Development notes
-----------------

After cloning the `official repository <https://github.com/mspronesti/qlearnkit>`__, create a virtual environment

.. code:: bash

   python3 -m venv .venv

and activate it

.. code:: bash

   source .venv/bin/activate

now you can install the requirements

.. code:: bash

   pip install -r requirements-dev.txt

now run the tests

.. code:: bash

   make test

Make sure to run

.. code:: bash

   pre-commit install

to set up the git hook scripts. Now ``pre-commit`` will run
automatically on ``git commit``!

.. |Python Versions| image:: https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue.svg?style=flat&logo=python&logoColor=white
   :target: https://www.python.org/
.. |License| image:: https://img.shields.io/github/license/mspronesti/qlearnkit
   :target: https://opensource.org/licenses/Apache-2.0
.. |Build| image:: https://github.com/mspronesti/qlearnkit/actions/workflows/build-and-test.yml/badge.svg
   :target: https://github.com/mspronesti/qlearnkit/blob/master/.github/workflows/build-and-test.yml
.. |Upload Python Package| image:: https://github.com/mspronesti/qlearnkit/workflows/Upload%20Python%20Package/badge.svg
   :target: https://pypi.org/project/qlearnkit/
.. |PypI Versions| image:: https://img.shields.io/pypi/v/qlearnkit
   :target: https://pypi.org/project/qlearnkit/#history
