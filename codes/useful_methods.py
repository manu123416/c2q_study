### ======================= ###
###  Useful helper methods  ###
### ======================= ###
import numpy as np
from PIL import Image
import time
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute, transpile
import matplotlib.pyplot as plt



### ============================ ###
###  Calculate input parameters  ###
### ============================ ###

# Generates P matrix as in Eq. 17
# Returns tuple of alpha and beta matrices from P, see Eqs. 18 and 19
def get_ab(x_in):
    x_new = np.reshape(x_in, (int(len(x_in)/2), 2))
    i_max = int(len(x_new))
    j_max = int(np.ceil(np.log2(len(x_in))))

    P = np.ndarray((i_max, j_max), dtype=complex)
    alpha = np.ndarray((i_max, j_max), dtype=complex)
    beta = np.ndarray((i_max, j_max), dtype=complex)
    for j in range(j_max):

        # TODO: this portion may have optimization potential
        for (i, x) in enumerate(x_new):
            if j == 0:
                p = np.power(np.linalg.norm(x), 2)

                if p == 0:
                    a = 1
                    b = 0
                else:
                    a = x[0] / np.linalg.norm(x)
                    b = x[1] / np.linalg.norm(x)
            elif i >= i_max / (2**j):
                p = 0
                a = 1
                b = 0
            else:
                p = P[2*i,j-1] + P[2*i+1,j-1]

                if p == 0:
                    a = 1
                    b = 0
                else:
                    a = np.sqrt(P[2*i,j-1] / p)
                    b = np.sqrt(P[2*i+1,j-1] / p)

            # This is purely done for readability
            P[i,j] = p
            alpha[i,j] = a
            beta[i,j] = b

    return (alpha, beta)


# Returns values of theta, phi, r, and t according to Eq. 20
def get_params(alpha, beta):
    alpha_mag = np.abs(alpha)
    alpha_phase = np.angle(alpha)
    beta_mag = np.abs(beta)
    beta_phase = np.angle(beta)

    with np.errstate(divide='ignore'):
        theta = 2*np.arctan(beta_mag/alpha_mag)
    phi = beta_phase - alpha_phase
    r = np.sqrt(alpha_mag**2 + beta_mag**2)
    t = beta_phase + alpha_phase

    return theta, phi, r, t

# Returns tuple of theta, phi, r, and t tensors given input data
def input_data(x_in):
    return get_params(*get_ab(x_in))

### ======================= ###
###  Generic helper methods  ###
### ======================= ###

# Generates random data corresponding to Eq. 16
def random_data(n_qubits, magnitude=False, use_complex_data=False, seed=None):
    if seed:
        np.random.seed(seed)

    n_states = 2**n_qubits
    x = np.random.rand(n_states)

    if use_complex_data:
        x = x + (np.random.rand(n_states) * 1j)
    else:
        x = x * 255

    mag = np.linalg.norm(x)
    x_in = x / mag

    if magnitude:
        return x_in, mag
    else:
        return x_in


# Imports image into state vector corresponding to Eq. 16
def import_image(filename, magnitude=False):
    im = Image.open(filename, 'r')



    n_rows, n_cols = im.size
    data = list(im.getdata())

    input_image_vec = np.reshape(data, n_rows*n_cols*3)

    mag = np.linalg.norm(input_image_vec)
    x_norm = input_image_vec / mag

    if magnitude:
        return n_rows, n_cols, x_norm, mag
    else:
        return n_rows, n_cols, input_image_vec


def run_circuit(qc, shots, backend):
    # print('Circuit depth:', qc.decompose().depth())
    # print('Circuit size:', qc.decompose().size())

    # print(qc.decompose().count_ops())

    job = execute(qc, backend, shots=shots)


    result = job.result()
    counts = result.get_counts(qc)
    time = result.time_taken
    # print(counts)
    # plot_histogram(counts)

    return counts, time

def get_data(data, shots, magnitude=1, num_qubits=None):
    if num_qubits:
        x_out = np.zeros(2**num_qubits)
    else:
        x_out = np.zeros(len(data))
    for _, (b, d) in enumerate(data.items()):
        x_out[int(b, 2)] = int(d)/shots
    return magnitude * np.sqrt(x_out)


def get_fidelity(x_in, x_out):
    dp = np.dot(x_in, x_out)
    fidelity = (np.abs(dp)) ** 2
    return fidelity