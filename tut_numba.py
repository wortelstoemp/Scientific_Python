'''
import numpy as np
import time
from numba import vectorize, cuda

#@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        an_array[x, y] += 1

def main():
    N = 320000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    
    start = time.time()
    C = VectorAdd(A, B)
    A = increment_a_2D_array(A)
    vector_add_time = time.time() - start

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))
    print("VectorAdd took for % seconds" % vector_add_time)

if __name__=='__main__':
    main()
    main()
'''
import numba
import math
import time

@numba.jit(nopython=True)
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)

start = time.time()
print(hypot(3.0, 4.0))
delta_time = time.time() - start
print("%f seconds" % delta_time)

start = time.time()
print(hypot.py_func(3.0, 4.0))
delta_time = time.time() - start
print("%f seconds" % delta_time)