from typing import Union
import pennylane as qml


class QmlMixin:
    """Mixin for models built on top of Pennylane (QML)"""

    _backend: Union[str, qml.Device]
    _n_qubits: int

    def _set_qml_backend(self,
                         backend: Union[str, qml.Device]):
        """
        Internal method to set a pennylane device according to its type

        Args:
            The backend to set. Can be a
            :class:`~pennylane.Device` or a string
            (valid name of the backend)

        """
        if isinstance(backend, qml.Device):
            n_wires = len(backend.wires)
            if n_wires != self._n_qubits:
                raise ValueError(
                    f"Invalid number of wires for backend {backend.name}. "
                    f"Expected {self._n_qubits}, got {n_wires}"
                )
            self._backend = backend
        else:
            # shots left as default (1000)
            self._backend = qml.device(backend, wires=self._n_qubits)

    @property
    def backend(self) -> qml.Device:
        return self._backend

    @backend.setter
    def backend(self, backend: Union[str, qml.Device]):
        self._backend = backend

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits: int):
        self._n_qubits = n_qubits
