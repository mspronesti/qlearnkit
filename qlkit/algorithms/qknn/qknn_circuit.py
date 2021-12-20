import logging
import numpy as np
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

logger = logging.getLogger(__name__)


def construct_circuit(state_vector_1: np.ndarray,
                      state_vector_2: np.ndarray) -> QuantumCircuit:
    """
    Constructs a swap test circuit
    Args:
        state_vector_1:
        state_vector_2:

    Returns:

    """
    if len(state_vector_1) != len(state_vector_2):
        raise ValueError("Input state vectors must have same length to"
                         "perform swap test. Lengths were:"
                         f"{len(state_vector_1)}"
                         f"{len(state_vector_2)}")

    size = int(math.log(len(state_vector_1), 2))

    q = QuantumRegister(2 * size + 1)
    c = ClassicalRegister(1)
    swaptest = QuantumCircuit(q, c)

    swaptest.repeat(1)
    swaptest.initialize(state_vector_1, range(1, size + 1))
    swaptest.initialize(state_vector_2, range(size + 1, 2 * size + 1))

    swaptest.h(0)
    for i in range(1, size + 1):
        swaptest.cx(i + size, i)
        swaptest.toffoli(0, i, i + size)
        swaptest.cx(i + size, i)

    swaptest.h(0)
    swaptest.barrier()

    swaptest.measure(range(1), range(1))
    return swaptest
