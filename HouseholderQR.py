#########################
# HouseholderQR.py
#
# Implements QR decomposition
# using Householder transformations
#
# Taylor J. Smith - CSCI 554 (Winter 2022)
#########################

import math
import numpy as np
import time

n = int(input("Dimension n? "))

A = np.random.randint(1, 10, size=(n,n))

print("A =\n", A)

#########################
# Our implementation
#########################

Q = np.identity(n)
R = A.copy()
R = R.astype(float)

start1 = time.time()
    
for k in range(0,n):
    x = R[k:, k]

    e1 = np.zeros(n - k)
    e1[0] = 1

    Vk = x + (np.sign(x[0]) * np.linalg.norm(x) * e1)
    Vk = Vk / np.linalg.norm(Vk)

    H = 2 * np.outer(Vk, np.transpose(Vk))

    Qk = np.identity(n)
    Qk[k:, k:] = Qk[k:, k:] - np.dot(Qk[k:, k:], H)

    Q = np.dot(Q, np.transpose(Qk))

    for j in range(k, n):
        R[k:, j] = R[k:, j] - 2 * np.dot(Vk, np.dot(np.transpose(Vk), R[k:, j]))

end1 = time.time()

print("       Q =\n", Q)
print("       R =\n", np.triu(R))
print("    Time =", end1 - start1)

#########################
# "Good" implementation
#########################

start2 = time.time()

goodQ, goodR = np.linalg.qr(A)

end2 = time.time()

print("\"Good Q\" =\n", goodQ)
print("\"Good R\" =\n", goodR)
print("    Time =", end2 - start2)
