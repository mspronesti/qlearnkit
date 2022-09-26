from typing import Union
import pennylane as qml


class QmlMixin:
    """Mixin for models built on top of Pennylane (QML)"""

    _device: Union[str, qml.Device]
    _n_qubits: int

    def _set_qml_device(self,
                        device: Union[str, qml.Device]):
        """
        Internal method to set a pennylane device according to its type

        Args:
            The backend to set. Can be a
            :class:`~pennylane.Device` or a string
            (valid name of the backend)

        """
        if isinstance(device, qml.Device):
            n_wires = len(device.wires)
            if n_wires != self._n_qubits:
                raise ValueError(
                    f"Invalid number of wires for backend {device.name}. "
                    f"Expected {self._n_qubits}, got {n_wires}"
                )
            self._device = device
        else:
            # shots left as default (1000)
            self._device = qml.device(device, wires=self._n_qubits)

    @property
    def device(self) -> qml.Device:
        return self._device

    @device.setter
    def device(self, backend: Union[str, qml.Device]):
        self._device = backend

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits: int):
        self._n_qubits = n_qubits
