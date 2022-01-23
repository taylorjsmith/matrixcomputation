#########################
# BackwardSub.py
#
# Implements backward substitution
# for solving an upper triangular
# system of linear equations
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
from scipy import linalg as splg
import sys
import time

k = int(input("Dimension k? "))

A = np.random.randint(1, 10, size=(k,k))
A = np.triu(A)

print("A =\n", A)

b = np.random.randint(1, 10, size=k)
x = b.astype(float)

print("b =\n", b)

#########################
# Our implementation
#########################

start1 = time.time()

for i in range(k-1, -1, -1):
    x[i] = b[i]
    for j in range(i+1, k):
        x[i] = x[i] - A[i][j]*x[j]
    if A[i][i] == 0:
        sys.exit("Error!")
    x[i] = x[i] / A[i][i]

end1 = time.time()

print("       x =\n", x)
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

x = splg.solve_triangular(A, b, lower=False)

end2 = time.time()

print("\"Good\" x =\n", x)
print("    Time =", end2 - start2)
