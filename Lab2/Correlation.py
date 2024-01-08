# -*- coding: cp1250 -*-

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

a = [1,2,3,4]
b = [4,3,2,1]

c = np.correlate(a,b,'full')
print(c)


A = np.fft.fft([0,0,0]+a)
B = np.fft.fft(b+[0,0,0])

C = np.conjugate(B)*A
c = np.fft.ifft(C)
print(np.real(c))

