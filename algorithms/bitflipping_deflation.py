import numpy as np
from .bitflipping import bitflipping
def bitflipping_deflation(tensor, number_of_components):
    (D, M, N) = tensor.shape
    U = np.zeros((D, number_of_components))
    V = np.zeros((M, number_of_components))
    for i in range(number_of_components):
        tensori = deflate(tensor, U, V)
        u, v = bitflipping(tensori)[0:2]
        U[:, i] = u
        V[:, i] = v
    return U, V
def deflate(tensor, U, V):
    (D, M, N) = tensor.shape
    tensor_def = np.zeros((D, M, N))
    Pu = np.eye(D) - U @ U.T
    Pv = np.eye(M) - V @ V.T
    for n in range(N):
        tensor_def[:, :, n] = Pu @ tensor[:, :, n] @ Pv
    return tensor_def