#########################
# Cholesky.py
#
# Implements Cholesky's method
# for decomposing a positive
# definite matrix
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
R = np.eye(n)

print("A =\n", A)

#########################
# Our implementation
#########################

start1 = time.time()

for i in range(0,n):
    R[i][i] = A[i][i]
    for k in range(0,i):
        R[i][i] = R[i][i] - R[k][i]*R[k][i]
    if R[i][i] <= 0:
        sys.exit("Error!")
    R[i][i] = math.sqrt(R[i][i])
    for j in range(i+1,n):
        R[i][j] = A[i][j]
        for k in range(0,i):
            R[i][j] = R[i][j] - R[k][i]*R[k][j]
        R[i][j] = R[i][j] / R[i][i]

end1 = time.time()

print("       R =\n", R)
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodR = splg.cholesky(A)

end2 = time.time()

print("\"Good\" R =\n", goodR)
print("    Time =", end2 - start2)
