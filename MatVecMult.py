#########################
# MatVecMult.py
#
# Implements matrix-vector multiplication
# in two ways (manually and using built-ins)
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import time

m = int(input("Dimension m? "))
n = int(input("Dimension n? "))

A = np.random.randint(1, 10, size=(m,n))
x = np.random.randint(1, 10, size=n)

print("A =\n", A)
print("x =\n", x)

#########################
# Our implementation
#########################

start1 = time.time()

b = 0
for i in range(0,m):
    bi = 0
    for j in range(0,n):
        bi = bi + A[i,j]*x[j]
    b = b + bi

end1 = time.time()

print("   Our b =", b)
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodb = np.sum(np.inner(A, x.T))

end2 = time.time()

print("\"Good\" b =", goodb)
print("    Time =", end2 - start2)
