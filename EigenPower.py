#########################
# EigenPower.py
#
# Implements the power method
# to find the dominant eigenvalue
# and eigenvector of a matrix
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import numpy as np
from scipy import linalg as splg
import time

n = int(input("Dimension n? "))
iters = int(input("Number of iterations? "))

A = np.random.randint(1, 10, size=(n,n))
v = np.random.randint(1, 10, size=n)

print("A =\n", A)
print("v =\n", v)

#########################
# Our implementation
#########################

start1 = time.time()

for i in range(iters):
    Av = np.dot(A, v)
    lam = abs(Av).max()
    v = Av / Av.max()

end1 = time.time()

print("   Dominant eigenvalue =", lam)
print("Associated eigenvector =", v)
print("                  Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodEigvals = splg.eigvals(A)

end2 = time.time()

print("    \"Good\" eigenvalues =", goodEigvals)
print("                  Time =", end2 - start2)