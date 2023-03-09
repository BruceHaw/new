import numpy as np
idxs_labels = np.array([[3, 2, 0, 4, 1],
                        [5, 8, 3, 1, 2]])

idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
idxs = idxs_labels[0,:]
print(idxs)