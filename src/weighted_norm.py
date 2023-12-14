import numpy as np

#-----------------------------------------------------------------------#
def weighted_norm(flux, perm, dim):
    if dim == 0:
        return 0

    inv_perm = [[1/k for k in perm[:, 0]]]
    for j in range(1, dim):
        inv_perm.append([1/k for k in perm[:, j]])
    inv_perm = np.asarray(inv_perm)

    norm = []
    num_cells = perm.shape[0]
    for i in range(num_cells):
        norm.append(np.square(flux[0, i])*inv_perm[0, i])
        for j in range(1, dim):
            norm[i] += np.square(flux[j, i])*inv_perm[j, i]

    return np.sqrt(np.asarray(norm))
