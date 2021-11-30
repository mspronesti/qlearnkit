from qiskit.aqua.algorithms.classifiers.qsvm._qsvm_abc import _QSVM_ABC


class _QKNN_Classifier(_QSVM_ABC):
    """
    This class inherits from  _QSVM_ABC provided by qiskit aqua
    which represents the base class for the binary and multiclass
    classifier. It represents the instance of the actual classifier.
    For instance, the Qiskit SVM has _QSVM_Binary or _QSVM_Multiclass
    """

    def __init__(self, qalgo):
        super().__init__(qalgo)

    def predict(self, data):
        """
        TODO: to be implemented
        Args:
            data:

        Returns:

        """
        pass

    def run(self):
        """
        TODO: to be implemented
        Returns:

        """
        pass
