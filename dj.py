import qiskit as q
import numpy as np
from math import pi
import time

#please watch sentdex's tutorial here for explanation:
#https://pythonprogramming.net/Deutsch-jozsa-hadamard-quantum-computer-programming-tutorial/
print("Quantum computer is starting")
start_time = time.time() #start time for quantum program

#ibm account code for quantum computer
#q.IBMQ.save_account("4842dc61d90a83c6c305c6ef6fc75c2aa65af5b26bd202989032ae14b7df56c2d9cc0e14ac1a3847ecb539b214cf9f08b894dfeb1f72ac291a2d57c19729d1b3")
q.IBMQ.load_account() #login

provider = q.IBMQ.get_provider("ibm-q") #get available quantum computers

backend = provider.get_backend("ibmq_santiago") #quantum computer

def balanced_black_box(qc):
    qc.cx(0,2)
    qc.cx(1,2)
    return qc

def constant_black_box(qc):
    return qc

qc = q.QuantumCircuit(3,2)
qc.h(0)
qc.h(1)
qc.h(2)
qc = balanced_black_box(qc)
qc.h(0)
qc.h(1)


