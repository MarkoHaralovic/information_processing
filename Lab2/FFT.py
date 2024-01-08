# Demostrating Fast Fourier Transform
import numpy as np
import scipy as sc
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

# x = np.random.random(1024)
# close = np.allclose(DFT_slow(x), np.fft.fft(x))


nuc2num = {}
nuc2num['A'] = 1
nuc2num['C'] = 2
nuc2num['G'] = 3
nuc2num['T'] = 4

#      0         1         2         3         4         5
#      012345678901234567890123456789012345678901234567890123456789
seq = 'GATTTGGGGTTCAAAGCAGTATCGATCAAATAGTAAATCCATTTGTTCAACTCACAGTTT'

#       0         1         2         3         4         5
#       012345678901234567890123456789012345678901234567890123456789
# seq1 = 'GATTTGGGGTTCAAAGCAGTA'
# seq2 = 'GATTTGGGCTTCAATGCTGTA'

#       0         1         2         3         4         5
#       012345678901234567890123456789012345678901234567890123456789
seq1 = 'GATTTGGG'
seq2 = 'CCGATCTA'

x1 = [nuc2num[s] for s in seq1[:20]]
x2 = [nuc2num[s] for s in seq2[:20]]

avg1 = sum(x1)/float(len(x1))
avg2 = sum(x2)/float(len(x2))

x1 = [x-avg1 for x in x1]
x2 = [x-avg2 for x in x2]

cor1 = np.correlate(x1, x2, "full")

X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)

X1_C = np.conjugate(X1)

cor_DFT = X1*X2

cor2 = np.fft.ifft(cor_DFT)
