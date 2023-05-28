import pennylane as qml
import numpy as np


dev_1 = qml.device("default.qubit", wires=4)
@qml.qnode(dev_1)
def ASymmetricDNAEncoder(X, div_2 = False):
    qml.Rot(phi =X[0], theta = X[1], omega = X[2], wires = 0)
    qml.Rot(phi =X[1], theta = X[2], omega = X[0], wires = 1)
    qml.Rot(phi =X[2], theta = X[0], omega = X[1], wires = 2)
    for i in range(3):
        qml.CNOT(wires = [i,3])
    for i in range(3):
        qml.CNOT(wires = [3,i])
    return qml.density_matrix(wires = [0,3])


dev_2 = qml.device("default.qubit", wires=4)
@qml.qnode(dev_2)
def SymmetricDNAEncoder(X, div_2 = False):
    qml.Rot(phi =X[0], theta = X[1], omega = X[2], wires = 0)
    qml.Rot(phi =X[2], theta = X[0], omega = X[1], wires = 1)
    qml.Rot(phi =X[1], theta = X[2], omega = X[0], wires = 2)
    for i in range(3):
        qml.CNOT(wires = [i,3])
    for i in range(3):
        qml.CNOT(wires = [3,i])
    return qml.density_matrix(wires = [0,3])

