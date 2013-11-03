import numpy as np
from numpy import linalg

A = np.array([(2, -1, 0), (-1, 2, -1), (0, -1, 2)])

w, v = linalg.eig(A)

print("The eigenvalues are:")
print w
print("and the eigenvectors are:")
print v

