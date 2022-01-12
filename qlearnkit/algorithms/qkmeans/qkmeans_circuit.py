import logging
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

logger = logging.getLogger(__name__)


def _map_function(x):
    r"""
    We map data feature values to :math:`\theta` and :math:`\phi` values using
    the following eqaution:

    .. math:: \phi = (x + 1) \frac{\pi}{2}

    where :math:`\phi` is the phase and :math:`\theta` the angle
    """
    return (x + 1) * np.pi / 2


def _map_features(input_point,
                  centroids,
                  n_centroids: int):
    r"""
    Map the input point and the centroids to :math:`\theta` and :math:`\phi` values
    via the :func:`_map_function` method.

    Args:
        input_point:
            Input point to map.
        centroids:
            Array of points to map.
        n_centroids:
            Number of centroids.
    Returns:
        Tuple containing input point and centroids mapped.
    """
    phi_centroids_list = []
    theta_centroids_list = []
    phi_input = _map_function(input_point[0])
    theta_input = _map_function(input_point[1])
    for i in range(0, n_centroids):
        phi_centroids_list.append(_map_function(centroids[i][0]))
        theta_centroids_list.append(_map_function(centroids[i][1]))
    return phi_input, theta_input, phi_centroids_list, theta_centroids_list


def construct_circuit(input_point: np.ndarray,
                      centroids: np.ndarray,
                      k: int) -> QuantumCircuit:
    """
    Apply a Hadamard to the ancillary qubit and our mapped data points.
    Encode data points using U3 gate.
    Perform controlled swap to entangle the state with the ancillary qubit
    Apply another Hadamard gate to the ancillary qubit.

    .. parsed-literal::

                    ┌───┐                   ┌───┐
            |0anc>: ┤ H ├────────────■──────┤ H ├────────M
                    └───┘            |      └───┘
                    ┌───┐   ┌────┐   |
            |0>: ───┤ H ├───┤ U3 ├───X──────────
                    └───┘   └────┘   |
                    ┌───┐   ┌────┐   |
            |0>: ───┤ H ├───┤ U3 ├───X──────────
                    └───┘   └────┘
    Args:
        input_point:
            Input point from which calculate the distance.
        centroids:
            Array of points representing the centroids to calculate 
            the distance to.
        k:
            Number of centroids.

    Returns:
        The quantum circuit created.
    """
    phi_input, theta_input, phi_centroids_list, theta_centroids_list = \
        _map_features(input_point, centroids, k)

    # We need 3 quantum registers, of size k one for a data point (input),
    # one for each centroid and one for each ancillary
    qreg_input = QuantumRegister(k, name='qreg_input')
    qreg_centroid = QuantumRegister(k, name='qreg_centroid')
    qreg_psi = QuantumRegister(k, name='qreg_psi')

    # Create a k bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(k, 'creg')

    # Create the quantum circuit containing our registers
    qc = QuantumCircuit(qreg_input, qreg_centroid, qreg_psi, creg, name='qc')

    for i in range(0, k):
        # Apply Hadamard
        qc.h(qreg_psi[i])
        qc.h(qreg_input[i])
        qc.h(qreg_centroid[i])

        # Encode new point and centroid
        qc.u(theta_input, phi_input, 0, qreg_input[i])
        qc.u(theta_centroids_list[i], phi_centroids_list[i], 0, qreg_centroid[i])

        # Perform controlled swap
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i])

        # Apply second Hadamard to ancillary
        qc.h(qreg_psi[i])

        # Measure ancillary
        qc.measure(qreg_psi[i], creg[i])

    return qc
