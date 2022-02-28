#########################
# GramSchmidt.py
#
# Implements the Gram-Schmidt process
# to solve least squares
# (and to compute a QR decomposition)
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import sys
import time

m = int(input("Dimension m? "))
n = int(input("Dimension n? "))

while m < n:
    print("m must be greater than or equal to n.")
    m = int(input("Dimension m? "))
    n = int(input("Dimension n? "))

V = np.random.randint(0,10,(m,n))
rk = np.linalg.matrix_rank(V)

# We want our matrix to have linearly independent columns
# If rank < number of columns, generate a new random matrix
# (We will rarely enter this while loop, as randomly-generated
#  matrices are very likely to have linearly independent columns)
while rk < n:
    V = np.random.rand(m,n)
    rk = np.linalg.matrix_rank(V)

print("V =\n", V)

#########################
# Our implementation
#########################

Q = np.copy(V.astype(float))

start1 = time.time()

for k in range(0,n):
    Vk = V[:,k]
    qhatk = Vk

    for i in range(0,k):
        rik = np.inner(Vk, Q[:,i])
        qhatk = qhatk - rik*Q[:,i]

    rkk = np.linalg.norm(qhatk)

    if rkk == 0:
        sys.exit("Columns v1 through vk are linearly dependent!")
    
    qk = (1 / rkk)*qhatk
    Q[:,k] = qk

end1 = time.time()

print("       Q =\n", Q)
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodQ, goodR = np.linalg.qr(V)

end2 = time.time()

print("\"Good Q\" =\n", goodQ)
print("    Time =", end2 - start2)
