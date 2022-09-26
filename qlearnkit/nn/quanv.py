from typing import Union
import qlearnkit.optionals as _optionals

if _optionals.HAS_PENNYLANE:
    # then qlearnkit has been installed
    # with the 'pennylane' option (or the env
    # already has both pennylane and torch)
    import pennylane as qml

    from pennylane.templates import (
        AngleEmbedding,
        BasicEntanglerLayers
    )
    from .qml_mixin import QmlMixin

    import torch
    from torch import Tensor
    from torch.nn import Module
    from torch.nn.functional import unfold

    from pennylane.qnn import TorchLayer as TorchConnector

else:
    from unittest.mock import Mock

    # allows importing the module, but
    # the instantiation will raise an Exception
    Module = Mock
    Tensor = Mock


@_optionals.HAS_PENNYLANE.require_in_instance
class Quanv2DLayer(Module, QmlMixin):
    """Quantum Convolution Layer for Pytorch Hybrid Architectures

    **References:**

    [1] Henderson et al.,
        `Quantum Long Short-Term Memory <https://arxiv.org/pdf/1904.04767.pdf>`_

    Args:
        n_qubits:
            int, The number of qubits of the quantum node.
            Default: 4.

        kernel_size:
            Size of the convolutional kernel.
            Default: (2,2).

        stride:
            Stride of the convolution.
            Default: 1

        out_channels:
             Number of channels produced by the quanvolution.
             Default: 4.

        n_layers:
            int, the number of layers of the variational
            quantum circuit.
            Default: 1.

        device:
            Can be a string representing the device name
            or a valid :class:`~pennylane.Device` having
            ``n_qubits`` wires.
            Default 'default.qubit'

    """

    def __init__(self,
                 n_qubits: int = 4,
                 kernel_size=(2, 2),
                 stride: int = 2,
                 out_channels: int = 4,
                 n_layers: int = 1,
                 device: Union[str, qml.Device] = 'default.qubit'
                 ) -> None:
        super(Quanv2DLayer, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        # calls the device setter
        # and makes sure the param
        # is legit
        self.device = device
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # create the quantum node
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = self._make_qnode(weight_shapes=weight_shapes)

    def quanvolve(self, inputs: Tensor, in_channels: int = 1):
        """
        Applies a single channel 2D quantum convolution (quanvolution)

        Args:
            inputs:
                one channel input batch
            in_channels:
                number of channel in the original input image

        Returns:

        """
        input_patches = unfold(inputs, kernel_size=self.kernel_size, stride=self.stride)
        s = input_patches.shape

        input_patches = input_patches.transpose(1, 2).reshape(s[0] * s[2], -1)
        convolved_patches = self.qnode(inputs=input_patches)\
            .view(s[0], s[2], -1)\
            .transpose(1, 2)

        h_out = (inputs.shape[2] - self.kernel_size[0]) // self.stride + 1
        w_out = (inputs.shape[3] - self.kernel_size[1]) // self.stride + 1

        out_shape = (
            inputs.shape[0],
            self.out_channels // in_channels,
            h_out,
            w_out
        )

        out = convolved_patches.view(out_shape)
        return out

    def _make_qnode(self, weight_shapes):
        @qml.qnode(self.device, interface='torch')
        def _qnode(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        return TorchConnector(_qnode, weight_shapes=weight_shapes)

    def forward(self, inputs: Tensor):
        """
        Applies the quantum convolution operation for the forward pass

        Args:
            inputs: :class:`~torch.Tensor`,
            input to be forwarded through the layer.

        Returns:
            Array containing model's prediction.
        """
        # unstack the different channels, apply convolutions, restack together
        return torch.cat([
            self.quanvolve(x.unsqueeze(1), in_channels=inputs.shape[1])
            for x in torch.unbind(inputs, dim=1)],
            # cat second arg
            1
        )
