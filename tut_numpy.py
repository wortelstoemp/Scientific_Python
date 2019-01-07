# ------------------------------------------------------------------------------
# Numpy cheatsheet
# ------------------------------------------------------------------------------
# USE NUMPY INSTEAD OF LISTS (FASTER)!!!
# For linear algebra use scipy.linalg instead of np.linalg!

# Numpy array = grid of values of SAME type, 
# index = tuple of positive ints. 
# number of dimensions =  rank or tensor order of the array.
# shape = tuple of ints giving size of array along each dimension.

# The plan
# ---------
# Know how to create arrays : array, arange, ones, zeros.
# Know the shape of the array with array.shape, 
# then use slicing to obtain different views of the array: array[::2],... 
# Adjust the shape of the array using reshape() or flatten it with ravel().
# Obtain a subset of the elements of an array and/or modify their values with masks.
# Know operations on arrays.
# For advanced use: master the indexing with arrays of ints, broadcasting. 

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

# Tiling (repeat a several times e.g. 2x along certain axes)
a = np.array([0, 1, 2])
b = np.tile(a, (2, 2))

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

# Elementwise operations
a = np.array([1, 2, 3, 4])
a + 1
b = np.ones(4) + 1
a - b
2 ** a
a * b   # Elementwise!!!
np.sin(a)
np.log(a)
np.exp(a)

np.triu(a)              # Upper triangle
np.tril(a)              # Lower triangle
a.T                     # Transpose

a == b                  # bool for each element
a > b           
np.array_equal(a, b)    # 1 bool
np.logical_and(a, b)
np.logical_or(a, b)
np.logical_xor(a, b)
np.logical_not(a, b)

np.exp(a)
np.log(a)
np.sin(a)


# Matrix multiplication
a = np.ones((3, 3))
b = a.dot(a)


# Reductions
x = np.array([1, 2, 3, 4])
np.sum(x)
x.sum()                 # alternative
x.min()
x.max()
x.argmin()              # index of minimum
x.argmax()              # index of maximum
x.trace()
x.mean()
np.median(x)
x.std()

x = np.array([[1, 1], [2, 2]])
x.sum(axis = 0)         # columns (first dimension)
x.sum(axis = 1)         # rows (second dimension)
x.sum(axis = -1)        # last axis

a = np.zeros((100, 100))
np.any(a != 0)
np.all(a == a)


# Broadcasting (transform arrays of different sizes to same size)
# uses the same principle as elementwise operations, but now for any shape
# http://www.scipy-lectures.org/_images/numpy_broadcasting.png


# Shape manipulation
a = np.array([[1, 2, 3], [4, 5, 6]])
a = a.ravel()                               # flatten to 1D array
b = a.flatten()                             # return a new flattened copy
b = a.reshape((2, 3))
b = a.reshape((2, -1))                      # -1 => inferred

# Adding a new axis
z = np.array([1, 2, 3])
z[:, np.newaxis]

# Resizing
a = np.arange(4)
a.resize((8,))

# Out-of-place sorting
a = np.array([[4, 3, 5], [1, 2, 1]])
b = np.sort(a, axis = 0)                    # sort each column
b = np.sort(a, axis = 1)                    # sort each row

# In-place sorting
a.sort()
print(a)

# Return array of sorted index order
a = np.array([4, 3, 1, 2])
indexes = np.argsort(a)
print(indexes)

# Loading data
# data = np.loadtxt("data/populations.txt")
# print(data)

# Loading data in binary
# data = np.load("pop.npy")

# Saving data
# np.savetxt("pop.txt", data)

# Saving data in binary
# data = np.ones((3, 3))
# np.save('pop.npy', data)

# Polynomials
p = np.poly1d([3, 2, -1])                   # 3x^2 + 2x - 1
p(0)
r = p.roots

# Casing
a = np.array([1.7, 1.2, 1.6])
b = a.astype(int)                           # <-- truncates to integer

# Rounding
a = np.array([1.2, 1.5, 1.6, 2.5, 3.5, 4.5])
b = np.around(a)
b                                           # still float !!!
c = np.around(a).astype(int)                
c                                           # casted to int

# Meshgrid
# x = np.array([1, 2, 3, 4])
# y = np.array([5, 6, 7])
# XX, YY = np.meshgrid(x, y)
# 1 7   2 7     3 7      4 7
# 1 6   2 6     3 6      4 6
# 1 5   2 5     3 5      4 5
# XX =  1 2 3 4     YY =    7 7 7 7
#       1 2 3 4             6 6 6 6
#       1 2 3 4             5 5 5 5


# TODO(tom): ogrid and mgrid