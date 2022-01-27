from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import Unroller


def to_basis_gates(qcircuit, basis=None):
    """Unrolls a given quantum circuit using basis gates
    """
    basis = ['u', 'cx'] if basis is None else basis

    unroller = Unroller(basis=basis)
    circuit_graph = circuit_to_dag(qcircuit)
    return dag_to_circuit(unroller.run(circuit_graph))
