from typing import Optional, Union
import qlearnkit.optionals as _optionals

if _optionals.HAS_PENNYLANE:
    # then qlearnkit has been installed
    # with the 'pennylane' option (or the env
    # already has both pennylane and torch)
    import pennylane as qml
    from pennylane.templates import AngleEmbedding

    from pennylane.qnn import TorchLayer as TorchConnector

    from torch.nn import Module
    from torch import Tensor

    from .qml_mixin import QmlMixin
else:
    from unittest.mock import Mock

    # allows importing the module, but
    # the instantiation will raise an Exception
    Module = Mock
    Tensor = Mock


@_optionals.HAS_PENNYLANE.require_in_instance
class QuantumAutoEncoder(Module, QmlMixin):
    """
    Hybrid Quantum AutoEncoder, miming pytorch's API and exploiting
    :class:`~pennylane.qnn.TorchLayer`.

    **References:**

    [1] Romero et al.,
        `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/abs/1612.02806>`_

    Args:
        n_qubits:
            int, the actual number of qubits used in the
            training. It's the `n` of the paper formalism.

        n_aux_qubits:
            int, the number of auxiliary qubits, used as
                ancillas in the swap test.

        n_latent_qubits:
            int, the number of qubits that can be discarded
            during the encoding.

        device:
            Can be a string representing the device name
            or a valid :class:`~pennylane.Device` having
            ``n_qubits`` wires
    """

    def __init__(self,
                 n_qubits: int = 4,
                 n_aux_qubits: int = 1,
                 n_latent_qubits: int = 1,
                 device: Optional[Union[str, qml.Device]] = 'default.qubit'
                 ) -> None:
        super(QuantumAutoEncoder, self).__init__()
        self.n_qubits = n_qubits + n_latent_qubits + n_aux_qubits

        self.n_aux_qubits = n_aux_qubits
        self.n_latent_qubits = n_latent_qubits

        self._set_qml_device(device)

        # define model parameters
        weight_shapes = {
            "params_rot_begin": (n_qubits * 3),
            "params_rot_ctrl": (n_qubits * (n_qubits - 1) * 3),
            "params_rot_end": (n_qubits * 3)
        }

        # create a quantum node layer interfaced via torch
        q_layer = qml.QNode(self._circuit,
                            self.device,
                            interface='torch')
        self.q_layer = TorchConnector(q_layer, weight_shapes)

    def _circuit(self, params_rot_begin, params_rot_ctrl,
                 params_rot_end, inputs):
        """Builds the circuit to be fed to the connector as a QML node"""
        self._embed_features(inputs[self.n_aux_qubits + self.n_latent_qubits:])

        wires = range(self.n_aux_qubits + self.n_latent_qubits,
                      self.n_qubits)
        i = 0
        for w in wires:
            qml.Rot(params_rot_begin[i],
                    params_rot_begin[i+1],
                    params_rot_begin[i+2],
                    wires=w)

            for w_rot in wires:
                if w != w_rot:
                    qml.CRot(params_rot_ctrl[i],
                             params_rot_ctrl[i+1],
                             params_rot_ctrl[i+2],
                             wires=[w, w_rot])

            qml.Rot(params_rot_end[i],
                    params_rot_end[i+1],
                    params_rot_end[i+2],
                    wires=w)

            i = i + 3
        # eventually, concatenate a swap test circuit
        self._swap_test()
        return [qml.probs(i) for i in range(self.n_aux_qubits)]

    def _embed_features(self, features):
        wires = range(self.n_aux_qubits + self.n_latent_qubits,
                      self.n_qubits)
        AngleEmbedding(features, wires=wires, rotation='X')

    def _swap_test(self):
        for i in range(self.n_aux_qubits):
            qml.Hadamard(wires=i)
        for i in range(self.n_aux_qubits):
            qml.CSWAP(wires=[i, i + self.n_aux_qubits, 2 * self.n_latent_qubits - i])
        for i in range(self.n_aux_qubits):
            qml.Hadamard(wires=i)

    def forward(self, x: Tensor):
        """
        Forward propagation pass of the Neural network.

        Args:
            x: :class:`~torch.Tensor`,
            input to be forwarded through the autoencoder

        Returns:
            Array containing model's prediction.
        """
        return self.q_layer(x)

    def __str__(self):
        return f"QuantumAutoencoder({self.n_qubits}, " \
               f"{self.n_aux_qubits}, {self.n_latent_qubits})"
