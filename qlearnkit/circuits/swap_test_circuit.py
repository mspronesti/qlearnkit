from qiskit import QuantumCircuit
from typing import Optional


class SwaptestCircuit(QuantumCircuit):
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

    A swap test circuit allows to measure the fidelity between two quantum
    states, which can be interpreted as a distance measure of some sort.

    In other words, given two quanutm states :math:`|\alpha\rangle, \ |\beta\rangle`,
    it measures how symmetric the state :math:`|\alpha\rangle \otimes |\beta\rangle` is
    """
    def __init__(self,
                 qc_state_1: QuantumCircuit,
                 qc_state_2: QuantumCircuit,
                 name: Optional[str] = None):
        n_total = qc_state_1.num_qubits + qc_state_2.num_qubits
        super().__init__(n_total + 1, 1, name=name)

        range_qc1 = [i + 1 for i in range(qc_state_1.num_qubits)]
        range_qc2 = [i + qc_state_1.num_qubits + 1 for i in range(qc_state_1.num_qubits)]

        self.compose(qc_state_1, range_qc1, inplace=True)
        self.compose(qc_state_2, range_qc2, inplace=True)

        # first apply hadamard
        self.h(0)
        # then perform controlled swaps
        for index, qubit in enumerate(range_qc1):
            self.cswap(0, qubit, range_qc2[index])
        # eventually reapply hadamard
        self.h(0)

        # Measurement on the auxiliary qubit
        self.measure(0, 0)
