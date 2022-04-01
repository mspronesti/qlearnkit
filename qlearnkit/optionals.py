"""Additional optional dependencies"""
from qiskit.utils import LazyImportTester

HAS_PENNYLANE = LazyImportTester(
    {
        "torch": (
            "cat",
            "tanh",
            "sigmoid",
            "nn",
            "zeros",
            "Tensor"
        ),
        "torch.nn": (
            "Linear",
            "Module",
            "Parameter",
        ),
        "pennylane": (
            "PauliZ",
            "templates",
            "qnn"
        ),
        "pennylane.templates": (
            "AngleEmbedding",
        ),
        "pennylane.qnn": (
            "TorchLayer",
        )
    },
    name="Pennylane",
    install="pip install 'qlearnkit[pennylane]'",
)
