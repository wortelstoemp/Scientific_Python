# Scipy

# Submodules
# scipy.cluster     Vector quantization / Kmeans
# scipy.constants   Physical and mathematical constants
# scipy.fftpack     Fourier transform
# scipy.integrate   Integration routines
# scipy.interpolate Interpolation
# scipy.io          Data input and output
# scipy.linalg      Linear algebra routines
# scipy.ndimage     n-dimensional image package
# scipy.odr         Orthogonal distance regression
# scipy.optimize	Optimization
# scipy.signal      Signal processing
# scipy.sparse      Sparse matrices
# scipy.spatial     Spatial data structures and algorithms (kd-trees,...)
# scipy.special     Any special mathematical functions
# scipy.stats       Statistics

# for solving PDE’s: use fipy or SfePy!

# import numpy as np
# from scipy import stats  # same for other sub-modules

import numpy as np  # scipy depends on numpy

# -------------------
# I/O of Matlab files
# -------------------
from scipy import io as spio

a = np.ones((3, 3))
spio.savemat("file.mat", {'a': a}) # .mat file is a dictionary
data = spio.loadmat("file.mat")
print(data['a'])


# -------------------
# Linear algebra
# -------------------
# Uses efficient BLAS, LAPACK implementations
from scipy import linalg

# Determinant
a = np.array([[1, 2], [3, 4]])
linalg.det(a)

# Inverse of square matrix
linalg.inv(a)

# SVD (singular value decomposition)
u, s, vt = linalg.svd(a)
a_reconstruction = u.dot(s).dot(vt)
np.allclose(a_reconstruction, a)

# Other decompositions (QR, LU, Cholesky, Schur)
# Linear solvers