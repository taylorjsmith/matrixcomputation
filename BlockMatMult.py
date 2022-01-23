#########################
# BlockMatMult.py
#
# Implements matrix-matrix multiplication
# using block matrices
#
# Assumptions:
# - A and X have power-of-two dimensions
# - A and X are partitioned into four blocks
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import time

def partitionMatrix(M, parts):
    M = np.array_split(M, parts, axis=0)
    newM = []

    for arr in M:
        arr = np.array_split(arr, parts, axis=1)
        newM = newM + [arr]

    return newM

k1 = 512
k2 = 512
p1 = 2
p2 = 2
A = np.random.randint(1, 10, size=(k1,k1))
X = np.random.randint(1, 10, size=(k2,k2))

print("A =\n", A)
print("X =\n", X)

#########################
# Our implementation
#########################

start1 = time.time()

newA = partitionMatrix(A, p1)
newX = partitionMatrix(X, p2)

B00 = np.matmul(newA[0][0], newX[0][0]) + np.matmul(newA[0][1], newX[1][0])
B01 = np.matmul(newA[0][0], newX[0][1]) + np.matmul(newA[0][1], newX[1][1])
B10 = np.matmul(newA[1][0], newX[0][0]) + np.matmul(newA[1][1], newX[1][0])
B11 = np.matmul(newA[1][0], newX[0][1]) + np.matmul(newA[1][1], newX[1][1])
B0 = np.concatenate((B00, B01), axis=1)
B1 = np.concatenate((B10, B11), axis=1)
B = np.concatenate((B0, B1), axis=0)

end1 = time.time()

print("   Our B =\n", B)
print("Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodB = np.matmul(A, X)

end2 = time.time()

print("\"Good\" B =\n", goodB)
print("Time =", end2 - start2)
