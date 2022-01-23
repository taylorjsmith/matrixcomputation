#########################
# MatMatMult.py
#
# Implements matrix-matrix multiplication
# in two ways (manually and using built-ins)
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import time

m = int(input("Dimension m? "))
n = int(input("Dimension n? "))
p = int(input("Dimension p? "))

A = np.random.randint(1, 10, size=(m,n))
X = np.random.randint(1, 10, size=(n,p))

print("A =\n", A)
print("X =\n", X)

#########################
# Our implementation
#########################

start1 = time.time()

B = np.zeros((m,p), dtype=int)
for i in range(0,m):
    for j in range(0,p):
        for k in range(0,n):
            B[i,j] = B[i,j] + A[i,k]*X[k,j]

end1 = time.time()

print("   Our B =\n", B)
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodB = np.matmul(A, X)

end2 = time.time()

print("\"Good\" B =\n", goodB)
print("    Time =", end2 - start2)
