# Calculating CFT the naive way
import numpy as np
def DFT_naive(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

x = np.random.random(1024)
close = np.allclose(DFT_naive(x), np.fft.fft(x))
print(close)

