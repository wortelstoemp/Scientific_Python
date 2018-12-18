# ------------------------------------------------------------------------------
# Numpy cheatsheet
# ------------------------------------------------------------------------------
# TODO: http://www.scipy-lectures.org/intro/numpy/array_object.html
# USE NUMPY INSTEAD OF LISTS (FASTER)!!!

# Numpy array = grid of values of SAME type, 
# index = tuple of positive ints. 
# number of dimensions =  rank or tensor order of the array.
# shape = tuple of ints giving size of array along each dimension.

import numpy as np

# Create arrays
a = np.array([1, 2, 3])             # Create a 1D array of type "numpy.ndarray"
b = np.arange(10) # 0 .. n-1
b = np.arange(1, 9, 2) # start, end (exclusive), step
c = np.linspace(0, 1, 6)   # start, end (inclusive), num float points

# 2D, 3D,... arrays
b = np.array([[1,2,3],[4,5,6]])     # Create a 2D array
print(b.shape)                      # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])    # Prints "1 2 4"

# Properties 
print(a.ndim)                       # Prints "1"
print(a.shape)                      # Prints "(3,)"
print(a[0], a[1], a[2])             # Prints "1 2 3"
a[0] = 5                            # Change an element of the array
print(a)                            # Prints "[5, 2, 3]"

# Initialization
a = np.empty((2, 2)                 # 2x2 matrix with garbage (no init)
a = np.zeros((2,2))                 # 2x2 zero matrix
print(a)
b = np.ones((1,2))                  # 1x2 one matrix
print(b)
c = np.full((2,2), 7)               # Create a constant array
print(c)
d = np.diag(np.array([1, 2, 3, 4])) # Create a 4x4 diagonal matrix
print(d)
e = np.eye(2)                       # Create a 2x2 identity matrix
print(e)

# Random numbers
np.random.seed(1234)                # Setting the random seed
r = np.random.rand((2,2))           # Create an array filled with random uniform values in [0, 1]
print(r)
n = np.random.randn(4)              # Gaussian
print(n)

# Data types
# int64, float64, int32, uint32, uint64, bool, complex128, S7 (string of max 7 chars)
a = np.zeros((2,2), dtype = float64) 

# Indexing and slicing: same as with Python lists

# TODO(tom): np.tile()

# Fancy indexing (masks)
a = np.random.randint(0, 21, 15)
mask = (a % 3 == 0)
m = a[mask]
a[mask] = -1

# Indexing with array of ints
a = np.arange(0, 100, 10)
print(a)
b = a[[2, 3, 2, 4, 2]]
print(b)

# TODO(tom): http://www.scipy-lectures.org/intro/numpy/operations.html