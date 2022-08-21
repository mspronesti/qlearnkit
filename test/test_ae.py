from qlearnkit.nn import QuantumAutoEncoder
import pytest
import torch


@pytest.mark.parametrize("n_qubits, n_latent_qubits, n_aux_qubits",
                         [(4, 2, 1),
                          (8, 2, 1),
                          (8, 4, 1),
                          (4, 1, 1),
                          (8, 3, 2)
                          ])
def test_autoencoder_output(n_qubits, n_latent_qubits, n_aux_qubits):
    x = torch.randn(2, n_qubits)

    ae = QuantumAutoEncoder(n_qubits, n_aux_qubits, n_latent_qubits)
    out = ae(x)

    assert out.shape == (2, 2 * n_aux_qubits)
