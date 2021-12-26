import logging
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

logger = logging.getLogger(__name__)


def construct_circuit(feature_vector_1: np.ndarray,
                      feature_vector_2: np.ndarray,
                      encoding_map,
                      name: str = None) -> QuantumCircuit:
    r"""
    Constructs a swap test circuit employing a controlled
    swap:

    .. parsed-literal::

                     ┌───┐       ┌───┐┌─┐
            q_0: ────┤ H ├─────■─┤ H ├┤M├
                 ┌───┴───┴───┐ │ └───┘└╥┘
            q_1: ┤ circuit-0 ├─X───────╫─
                 ├───────────┤ │       ║
            q_2: ┤ circuit-1 ├─X───────╫─
                 └───────────┘         ║
            c: 1/══════════════════════╩═
                                       0


    where feature_vector_1 = [1,0], feature_vector_2 = [0, 1]

    A swap test circuit allows to measure the fidelity between two quantum
    states, which can be interpreted as a distance measure of some sort.
    In other words, given two quanutm states :math:`|\alpha\rangle, \ |\beta\rangle`
    it measures how symmetric the state :math:`|\alpha\rangle \otimes |\beta\rangle` is

    Args:
        feature_vector_1:
            first feature vector
        feature_vector_2:
            second feature vector
        encoding_map:
            the mapping to quantum state to
            extract a :class:`~qiskit.QuantumCircuit`
        name:
            the (optional) name of the circuit

    Returns:
        swap test circuit

    """
    if len(feature_vector_1) != len(feature_vector_2):
        raise ValueError("Input state vectors must have same length to"
                         "perform swap test. Lengths were:"
                         f"{len(feature_vector_1)}"
                         f"{len(feature_vector_2)}")

    if encoding_map is None:
        raise ValueError("encoding map must be specified to construct"
                         "swap test circuit")

    qc_1 = encoding_map.circuit(feature_vector_1)
    qc_2 = encoding_map.circuit(feature_vector_2)

    n_total = qc_1.num_qubits + qc_2.num_qubits
    range_qc1 = [i + 1 for i in range(qc_1.num_qubits)]
    range_qc2 = [i + qc_1.num_qubits + 1 for i in range(qc_2.num_qubits)]

    # Constructing the swap test circuit
    qc_swap = QuantumCircuit(n_total + 1, 1, name=name)
    qc_swap.append(qc_1, range_qc1)
    qc_swap.append(qc_2, range_qc2)

    # Swap Test

    # first apply hadamard
    qc_swap.h(0)
    # then perform controlled swaps
    for index, qubit in enumerate(range_qc1):
        qc_swap.cswap(0, qubit, range_qc2[index])
    # eventually reapply hadamard
    qc_swap.h(0)

    # Measurement on the auxiliary qubit
    qc_swap.measure(0, 0)
    return qc_swap
