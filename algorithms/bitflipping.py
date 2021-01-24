import numpy as np
from utils import *
def bitflipping(data_tensor, *init_arg, tolerance = 1e-4):
    (D, M, N) = data_tensor.shape
    if len(init_arg) == 0:
        aux_bin_vector = np.sign(np.random.randn(N, ))
    else:
        aux_bin_vector = init_arg[0]
    X = xmatrix(data_tensor)    
    metric_evolution = [metric_of_b(X, aux_bin_vector)]
    I = np.eye(N)
    while True:
        val = np.zeros((N, 1))
        for n in range(N):
            bn = aux_bin_vector -2 * aux_bin_vector[n] * I[:, n]
            val[n] = metric_of_b(X, bn)
        met = np.max(val)
        if met - metric_evolution[-1] > tolerance:
            idx = np.argmax(val)
            aux_bin_vector = aux_bin_vector -2 * aux_bin_vector[idx] * I[:, idx]
            metric_evolution.append(met)
        else:
            U, S, Vt = np.linalg.svd(X @ np.kron(aux_bin_vector[:, None], np.eye(M)))
            u = U[:, 0]
            v = Vt[0, :]
            break
    return u, v, aux_bin_vector, metric_evolution