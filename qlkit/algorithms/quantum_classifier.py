import logging
import time
from typing import Union, Optional

import numpy as np
from abc import abstractmethod
import qiskit
from qiskit.providers import JobStatus, BaseBackend
from qiskit.qobj import Qobj
from qiskit.result import Result
from sklearn.base import ClassifierMixin, TransformerMixin


logger = logging.getLogger(__name__)


class QuantumClassifier(ClassifierMixin, TransformerMixin):
    def __init__(self,
                 encoding_map,
                 backend: BaseBackend,
                 shots: int = 1024,
                 optimization_level: int = 1):
        """

        Args:
            encoding_map:
                        map to classical data to quantum states.
                        This class does not impose any constraint on it. It
                        can either be a custom encoding map or a qiskit FeatureMap
            backend:
                the qiskit backend to do the compilation & computation on
            shots:
                number of repetitions of each circuit, for sampling. Default: 1024
            optimization_level:
                level of optimization to perform on the circuits.
                Higher levels generate more optimized circuits,
                at the expense of longer transpilation time.
                        0: no optimization
                        1: light optimization
                        2: heavy optimization
                        3: even heavier optimization
                If None or invalid value, level 1 will be chosen as default.
                Notice: a high optimization level requires way more time of
                        computation
        """
        self.encoding_map = encoding_map
        self.backend = backend
        self.shots = shots

        self.optimization_level = optimization_level \
            if optimization_level in range(0, 4) else 1

        self.X_train = np.asarray([])
        self.y_train = np.asarray([])
        self.qcircuits = None

    def fit_transform(self, X, y=None, **fit_params):
        return X

    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray):
        """
        Fits the model using X as training dataset
        and y as training labels
        Args:
            X_train: training dataset
            y_train: training labels

        """
        raise NotImplementedError("Must have implemented this.")

    @abstractmethod
    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the labels associated to the
        unclassified data X_test
        Args:
            X_test: the unclassified data

        Returns:
            the labels associated to X_test
        """
        raise NotImplementedError("Must have implemented this.")

    @abstractmethod
    def _create_circuits(self,
                         X_train: np.ndarray,
                         X_test: np.ndarray):
        """
        Creates the quantum circuit(s) to perform
        the classification process
        Args:
            X_test: input data to be classified

        Returns:
            quantum circuits
        """
        raise NotImplementedError("Must have implemented this.")

    def execute(self, X_test) -> Union[Optional[Result]]:
        """
        Executes the given quantum circuit
        Args:
            X_test: the unclassified input data

        Returns:
            the execution results
        """
        logger.info("Executing circuits...")
        self.qcircuits = self._create_circuits(self.X_train, X_test)

        # Instead of transpiling and assembling the quantum object
        # and running the backend, we call qiskit.execute that does
        # it at once a very efficient way
        job = qiskit.execute(
            self.qcircuits,
            self.backend,
            optimization_level=self.optimization_level,
            shots=self.shots
        )

        job.result()
        while not job.status() == JobStatus.DONE:
            if job.status() == JobStatus.CANCELLED:
                logger.info("Job cancelled...")
                break
            logger.info("Waiting for job to complete...")
            # this is not optimized
            # a condition variable would do this
            # in an actual efficient way, but we don't
            # have control on the job, then we can't explicitly
            # call the "cv.signal" method
            time.sleep(5)

        if job.status() == JobStatus.DONE:
            logger.info("Job completed!")
            return job.result()
        else:
            logger.error("Job not completed, errors occurred. Job status is: {}".format(job.status))
            return None

