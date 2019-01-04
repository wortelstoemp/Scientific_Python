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
# Linear algebra (scipy.linalg)
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
# TODO: https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg


# ------------------------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------------------------
# TODO: http://www.scipy-lectures.org/intro/scipy.html#interpolation-scipy-interpolate

# Faked experimental data
times = np.linspace(0, 1, 10)   # start, end (inclusive), num float points
noise = (np.random.random(10)*2 - 1) * 1e-1
measurements = np.sin(2 * np.pi * times) + noise

# Interpolation (interp1d for 1D arrays, or interp2d for 2D arrays)
from scipy.interpolate import interp1d
linear_interp = interp1d(times, measurements)
cubic_interp = interp1d(times, measurements, kind='cubic')

# Evaluate interpolation at times of interest
linear_interp = interp1d(times, measurements)
interpolation_times = np.linspace(0, 1, 50)
results = linear_interp(interpolation_times)


# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------
# TODO: http://www.scipy-lectures.org/advanced/mathematical_optimization/index.html#mathematical-optimization
from scipy import optimize

# Curve fitting (find parameters)
def test_func(x, a, b):
    return a * np.sin(b * x)    # a and b are params (amplitude and period)!!!

x_data = np.linspace(-5, 5, num=50)
y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)

params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])


# Local minimum of function
# minimize() works in general with x multidimensionsal!!!
# If 1D function, you can just use scipy.optimize.minimize_scalar()
def f(x):
    return x**2 + 10*np.sin(x)

starting_point = 0
result = optimize.minimize(f, x0 = starting_point, method="L-BFGS-B")
print(result.x)

# Global minimum
#   Basin-hopping is a two-phase method that combines a global 
#   stepping algorithm with local minimization at each step.
#   Other algorithms: shgo, dual_annealing, brute (= bad...)
starting_point = 0
result = optimize.basinhopping(f, x0 = starting_point, method="L-BFGS-B")
print(result.x)

# Constraints (vector variable x in []-interval)
result = optimize.minimize(f, x0 = 1, bounds = ((0, 10), ))


# Finding roots of function
root = optimize.root(f, x0 = 1)
print(root.x)


# ------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------
# TODO: http://www.scipy-lectures.org/packages/statistics/index.html#statistics 
# Mean
samples = np.random.normal(size = 1000)
np.mean(samples)

# Median (insensitive to outliers)
samples = np.random.normal(size = 1000)
np.median(samples) 

# Percentiles (or CDF)
# p'th percentile is the point where the CDF reaches p/100
stats.scoreatpercentile(samples, 90)

# Histogram and PDF's
# TODO: http://www.scipy-lectures.org/intro/scipy.html#distributions-histogram-and-probability-density-function

# T-Test test if means of 2 sets of observations (distributions) are close
# T value: difference between the processes
# p value: probability of 2 processes being equal
a = np.random.normal(0, 1, size = 100)
b = np.random.normal(1, 1, size = 10)
stats.ttest_ind(a, b) # gives T, p values


# ------------------------------------------------------------------------------
# Numerical integration
# ------------------------------------------------------------------------------
from scipy.integrate import quad

def f(x):
    return np.sin(x)

a = 0
b = np.pi/2
result, error_estimate = quad(f, a, b)

# Other schemes: scipy.integrate.fixed_quad(), 
# scipy.integrate.quadrature(), scipy.integrate.romberg()

# ------------------------------------------------------------------------------
# Integrating first-order ODE's (Ordinary Differential Equations)
# of form: dy/dt = func(y, t, ...)  [or func(t, y, ...)]
# ------------------------------------------------------------------------------
def func(y, t):
    return -2 * y

from scipy.integrate import odeint

times = np.linspace(0, 4, 40)
y = odeint(func, y0=1, t=times)


# ------------------------------------------------------------------------------
# Fast Fourier Transforms (use SciPy and NOT Numpy)
# ------------------------------------------------------------------------------
# from scipy import fftpack
# fft_signal = fftpack.fft(sig)
# frequencies = fftpack.fftfreq(signal.size, d=time_step) 

# ifft_signal = scipy.fftpack.ifft(fft_signal)


# ------------------------------------------------------------------------------
# Signal processing (1D signals or nD signals too)
# ------------------------------------------------------------------------------
# TODO: http://www.scipy-lectures.org/intro/scipy.html#signal-processing-scipy-signal
from scipy import signal

count = 100
t = np.linspace(0, 5, count)
x = np.sin(t)

# Resample a signal to another n number of samples (by using FFT)
n = 25
x_resampled = signal.resample(x, n)

# Detrending: remove linear tendency of a signal (make flat)
x_detrended = signal.detrend(x)

# Filtering 
# scipy.signal.medfilt(), scipy.signal.wiener(), IIR filters,...
# See https://docs.scipy.org/doc/scipy/reference/signal.html#module-scipy.signal

# Spectrogram (frequency over time windows)
# f, t, spectrogram = signal.spectrogram(x)

# Power Spectrum  Density (PSD)
# t, psd = scipy.signal.welch(x)


# ------------------------------------------------------------------------------
# Image processing (2D signals) = [y, x]
# ------------------------------------------------------------------------------
from scipy import ndimage

# Load a test image
from scipy import misc
face = misc.face(gray=True)
face = face[:512, -512:]

# Geometrical transformations
shifted_face = ndimage.shift(face, (50, 50))
shifted_face2 = ndimage.shift(face, (50, 50), mode="nearest")
rotated_face = ndimage.rotate(face, 30)
cropped_face = face[50:-50, 50:-50]
zoomed_face = ndimage.zoom(face, 2)

# Noisy image
import numpy as np
noisy_face = np.copy(face).astype(np.float)
noisy_face += face.std() * 0.5 * np.random.standard_normal(face.shape)

# Filters (scipy.ndimage.filters & scipy.signal)
blurred_face = ndimage.gaussian_filter(noisy_face, sigma=3)
median_face = ndimage.median_filter(noisy_face, size=5)
from scipy import signal
wiener_face = signal.wiener(noisy_face, (5, 5))

# Mathematical Morphology (erosion, dilation, opening, closing)
#   Opening: erosion, then dilation => removes small objects, smooths corners
#   Closing: dilation, then erosion => fills small holes
#   
#   First create binary mask from image, then do morphology on it and apply mask to image

a = np.zeros((7, 7), dtype=np.int)
a[1:6, 2:5] = 1

elem = ndimage.generate_binary_structure(2, 1).astype(np.int) # 3x3 "+ cross of 1s (True)"
elem2 = ndimage.generate_binary_structure(2, 2).astype(np.int) # 3x3 matrix of 1s

b = ndimage.binary_erosion(a, structure=elem).astype(a.dtype)
b = ndimage.binary_dilation(a, structure=elem).astype(a.dtype)
b = ndimage.binary_opening(a, structure=elem).astype(np.int)
b = ndimage.binary_closing(a, structure=elem).astype(np.int)

# Connected components
# TODO: http://www.scipy-lectures.org/intro/scipy.html#image_manipulation
