from typing import Optional, Union

try:
    import pennylane as qml
    from pennylane import PauliZ
    from pennylane.qnn import TorchLayer as TorchConnector

    import torch
    import torch.nn as nn
    from torch.nn import Module
    from torch import Tensor

    # whether qlearnkit has been installed
    # with the 'pennylane' option (or if the env
    # already has both pennylane and torch)
    _PENNYLANE_INSTALL = True
except ImportError:
    from unittest.mock import Mock
    # allows importing the module, but
    # the instantiation will raise an ImportError.
    # See the constructor of the following class
    Module = Mock
    Tensor = Mock

    _PENNYLANE_INSTALL = False


def H_layer(n_qubits):
    """
    Layer of single-qubit Hadamard gates.
    """
    for w in range(n_qubits):
        qml.Hadamard(wires=w)


def RY_layer(feats):
    """
    Layer of parametrized qubit rotations around the y axis.
    """
    for w, xi in enumerate(feats):
        qml.RY(xi, wires=w)


def entangling_layer(n_qubits):
    """
    Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # loop over even indices
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    # loop over odd indices
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


class QLongShortTermMemory(Module):
    r"""
    Hybrid Quantum Long Short Term Memory, miming pytorch's API and exploiting
    :class:`~pennylane.qnn.TorchLayer`.

    Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    **References:**

    [1] Chen et al.,
        `Quantum Long Short-Term Memory <https://arxiv.org/pdf/2009.01783.pdf>`_

    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: Optional[int] = 1,
                 n_qubits: Optional[int] = 4,
                 batch_first: Optional[bool] = True,
                 backend: Optional[Union[str, qml.Device]] = 'default.qubit',
                 ):
        """

        Args:
            input_size:
                the number of expected features in the input x
            hidden_size:
                the number of features in the hidden state h
            n_layers:
                the number of recurrent layers
            n_qubits:
                the number of qubits
            batch_first:
                if ``True``, then the input and output tensors are
                provided as (batch, seq, feature) instead of (seq, batch, length)
           backend:
                Can be a string representing the backend name
                or a valid :class:`~pennylane.Device` having
                ``n_qubits`` wires

        """
        if not _PENNYLANE_INSTALL:
            raise ImportError(
                "Module qlearnkit.nn.QLongShortTermMemory"
                "depends on pennylane and torch. "
                "Please install those or reinstall qlearnkit "
                "running `pip install qlearnkit[pennylane]`"
            )
        super(QLongShortTermMemory, self).__init__()

        self.input_size = input_size
        self.num_qubits = n_qubits
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.batch_first = batch_first

        self._set_qml_backend(backend)
        # classical layers
        self.clayer_in = nn.Linear(input_size + hidden_size, n_qubits)
        self.clayer_out = nn.Linear(n_qubits, hidden_size)

        # quantum layers
        self.q_layers = {}
        self._construct_quantum_layers()

    def _construct_quantum_network(self, inputs, weights):
        """
        Constructs the variational quantum circuit
        as described in https://arxiv.org/pdf/2009.01783.pdf
        """
        # Reshape weights
        q_weights = weights.reshape(self.num_layers, self.num_qubits)
        # Start from state |+> , unbiased w.r.t. |0> and |1>
        H_layer(self.num_qubits)

        # Embed features in the quantum node
        RY_layer(inputs)

        # Sequence of trainable variational layers
        for k in range(self.num_layers):
            entangling_layer(self.num_qubits)
            RY_layer(q_weights[k])

        # Retrieve expectation values in the Z basis
        return [
            qml.expval(qml.PauliZ(wires=w))
            for w in range(self.num_qubits)
        ]

    def _construct_quantum_layers(self):
        for layer_name in ['input', 'update', 'forget', 'output']:
            layer = qml.QNode(self._construct_quantum_network,
                              self.backend,
                              interface='torch')
            weight_shapes = {"weights": (self.num_layers, self.num_qubits)}

            self.q_layers[layer_name] = TorchConnector(layer, weight_shapes)

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
            if n_wires != self.num_qubits:
                raise ValueError(
                    f"Invalid number of wires for backend {backend.name}. "
                    f"Expected {self.num_qubits}, but got {n_wires}"
                )
            self.backend = backend
        else:
            # shots left as default (1000)
            self.backend = qml.device(backend, wires=self.num_qubits)

    def forward(self,
                input: Tensor,
                hx: Optional[Tensor] = None):
        r"""
        Forward propagation pass of the Recurrent Neural network using a LSTM cell

        Args:
            input: the input at time :math:`t`
            hx: Tensor containing the hidden state
                    :math:`h(t)` and the cell state :math:`c(t)` at time :math:`t`
        Returns:
            the hidden sequence at time :math:`t+1` and the new couple
            :math:`h(t+1)`, :math:`c(t+1)`.

        """
        if self.batch_first:
            batch, seq, _ = input.size()
        else:
            seq, batch, _ = input.size()

        hidden_seq = []
        if hx is None:
            h_t = torch.zeros(batch, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch, self.hidden_size)  # cell state
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            h_t, c_t = hx.detach()

        for t in range(seq):
            # get features from the t-th element,
            # for all batch entries
            x_t = input[:, t, :]

            # concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            # for each time step `t` we compute the forget, input, update and output gates
            # using 3 sigmoid layers and a hyperbolic tangent layer.
            # forget
            f_t = torch.sigmoid(self.clayer_out(self.q_layers['forget'](y_t)))
            # input
            i_t = torch.sigmoid(self.clayer_out(self.q_layers['input'](y_t)))
            # update
            g_t = torch.tanh(self.clayer_out(self.q_layers['update'](y_t)))
            # output
            o_t = torch.sigmoid(self.clayer_out(self.q_layers['output'](y_t)))

            # eventually, the hidden state and the cell state are evaluated
            # (see RNN architecture)
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # update hidden seq
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


