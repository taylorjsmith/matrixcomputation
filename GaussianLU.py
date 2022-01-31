#########################
# GaussianLU.py
#
# Uses Gaussian elimination
# to compute the LU decomposition
# of a matrix
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
from scipy import linalg as splg
import sys
import time

n = int(input("Dimension n? "))

A = np.random.randint(1, 10, size=(n,n))
A = np.matmul(A.transpose(), A)
A = A.astype(float)

print("   A =\n", A)

L = np.eye(n)
U = np.copy(A)

#########################
# Our implementation
#########################

start1 = time.time()

for k in range(0,n-1):
    for i in range(k+1,n):
        if A[k][k] == 0:
            sys.exit("Error!")
        L[i][k] = U[i][k] / U[k][k]
        for j in range(k+1,n):
            U[i][j] = U[i][j] - L[i][k]*U[k][j]

end1 = time.time()

print("   L =\n", L)
print("   U =\n", np.triu(U))
print("Time =", end1 - start1)

#########################
# "Good" implementation
#########################

print("\nNOTE: SciPy's implementation of LU factorization")
print("      uses an additional permutation matrix P.")
print("      This sometimes results in the SciPy matrices")
print("      differing from our computed matrices up to permuted rows.\n")

start2 = time.time()

P, goodL, goodU = splg.lu(A)

end2 = time.time()

print("        P =\n", P)
print(" \"Good\" L =\n", goodL)
print(" \"Good\" U =\n", goodU)
print("     Time =", end2 - start2)