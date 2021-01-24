import numpy as np
def metric_of_b(X, aux_bin_vec):
    (D, MN) = X.shape
    N = aux_bin_vec.shape[0]
    M = int(MN / N)
    if aux_bin_vec.ndim == 1: aux_bin_vec = aux_bin_vec[:, None]
    Z = X @ np.kron(aux_bin_vec, np.eye(M))
    return np.linalg.svd(Z)[1].flatten('F')[0]