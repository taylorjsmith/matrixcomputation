#########################
# GaussianSolve.py
#
# Implements Gaussian elimination
# for solving a system of
# linear equations, and solves the system
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import sys
import time

def BackwardSub(A, b):
    x = b.astype(float)
    k = b.size

    for i in range(k-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, k):
            x[i] = x[i] - A[i][j]*x[j]
        if A[i][i] == 0:
            sys.exit("Error!")
        x[i] = x[i] / A[i][i]
    
    return x

n = int(input("Dimension n? "))

A = np.random.randint(1, 10, size=(n,n))
A = np.matmul(A.transpose(), A)
A = A.astype(float)
copyA = np.copy(A)

print("A =\n", A)

b = np.random.randint(1, 10, size=n)
b = b.astype(float)
copyb = np.copy(b)

print("b =\n", b)

#########################
# Our implementation
#########################

###########
# Matrix A
###########

start1 = time.time()

for k in range(0,n-1):
    for i in range(k+1,n):
        if A[k][k] == 0:
            sys.exit("Error!")
        A[i][k] = A[i][k] / A[k][k]
        for j in range(k+1,n):
            A[i][j] = A[i][j] - A[i][k]*A[k][j]

###########
# Vector b
###########

for k in range(0,n-1):
    for i in range(k+1,n):
        b[i] = b[i] - A[i][k]*b[k]

###########
# Solve for x
###########

x = BackwardSub(A, b)

end1 = time.time()

print("Reduced A =\n", A)
print("Reduced b =\n", b)
print("        x =\n", x)
print("     Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

x = np.linalg.solve(copyA, copyb)

end2 = time.time()

print(" \"Good x\" =\n", x)
print("     Time =", end2 - start2)