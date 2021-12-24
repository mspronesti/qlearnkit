import logging
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

logger = logging.getLogger(__name__)


def construct_circuit(state_vector_1: np.ndarray,
                      state_vector_2: np.ndarray,
                      name: str = None) -> QuantumCircuit:
    r"""
    Constructs a slightly modified swap test circuit employing a Toffoli
    swap

    .. parsed-literal::

                         ┌───┐                 ┌───┐ ░ ┌─┐
            q1_0: ───────┤ H ├──────────────■──┤ H ├─░─┤M├
                  ┌──────┴───┴──────┐┌───┐  │  ├───┤ ░ └╥┘
            q1_1: ┤ Initialize(1,0) ├┤ X ├──■──┤ X ├─░──╫─
                  ├─────────────────┤└─┬─┘┌─┴─┐└─┬─┘ ░  ║
            q1_2: ┤ Initialize(0,1) ├──■──┤ X ├──■───░──╫─
                  └─────────────────┘     └───┘      ░  ║
            c1: 1/══════════════════════════════════════╩═
                                                        0

    where state_vector_1 = [1,0], state_vector_2 = [0, 1]

    A swap test circuit allows to measure the fidelity between two quantum
    states, which can be interpreted as a distance measure of some sort.
    In other words, given two quanutm states :math:`|\alpha\rangle, \ |\beta\rangle`
    it measures how symmetric the state :math:`|\alpha\rangle \otimes |\beta\rangle` is

    Args:
        state_vector_1: first state
        state_vector_2: second state
        name: the (optional) name of the circuit

    Returns:
        swap test circuit

    Note:
        state vectors must be normalized

    """
    if len(state_vector_1) != len(state_vector_2):
        raise ValueError("Input state vectors must have same length to"
                         "perform swap test. Lengths were:"
                         f"{len(state_vector_1)}"
                         f"{len(state_vector_2)}")

    size = int(np.log2(len(state_vector_1)))

    q = QuantumRegister(2 * size + 1)
    c = ClassicalRegister(1)
    swaptest = QuantumCircuit(q, c, name=name)

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

