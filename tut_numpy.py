# ------------------------------------------------------------------------------
# Numpy cheatsheet
# ------------------------------------------------------------------------------

# USE NUMPY INSTEAD OF LISTS (FASTER)!!!

# Numpy array = grid of values of SAME type, 
# indexed by a tuple of positive integers. 
# The number of dimensions is the rank or tensor order of the array.
# The shape of an array is a tuple of integers giving the size of the array along each dimension.

import numpy as np

a = np.array([1, 2, 3])             # Create a 1D array of type "numpy.ndarray"
print(a.shape)                      # Prints "(3,)"
print(a[0], a[1], a[2])             # Prints "1 2 3"
a[0] = 5                            # Change an element of the array
print(a)                            # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])     # Create a 2D array
print(b.shape)                      # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])    # Prints "1 2 4"


# Initialization
a = np.zeros((2,2))                 # 2x2 zero matrix
print(a)

b = np.ones((1,2))                  # 1x2 one matrix
print(b)

c = np.full((2,2), 7)               # Create a constant array
print(c)

e = np.eye(2)                       # Create a 2x2 identity matrix
print(e)

r = np.random.random((2,2))         # Create an array filled with random values
print(r)