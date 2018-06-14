# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:23:26 2018

@author: drr342
"""

from __future__ import print_function, division
from time import monotonic
import numpy as np

INPUT_DIM = 50176
L1_DIM = 4000
L2_DIM = 1000

def init(i, j):
	return 0.5 + ((i + j) % 50 - 30) / 50.0

def matrix_product (A, B, C) :
	for i in range(len(A)) :
		for j in range(len(B[0])) :
			C[i][j] = sum(map(lambda a, b: float(a) * float(b), A[i], [r[j] for r in B]))

def relu (v) :
	v = [list(map(max, v[0], [0] * len(v[0])))]


'''''''''''''''''''''''''''''
C2
'''''''''''''''''''''''''''''
x0 = [[init(i, j) for j in range(INPUT_DIM) for i in range(1)]]
w0 = [[init(i, j) for j in range(L1_DIM)] for i in range(INPUT_DIM)]
w1 = [[init(i, j) for j in range(L2_DIM)] for i in range(L1_DIM)]
z0 = [[0] * L1_DIM]
z1 = [[0] * L2_DIM]

start = monotonic()
matrix_product(x0, w0, z0)
relu(z0)
matrix_product(z0, w1, z1)
relu(z1)
s = sum(z1[0])
end = monotonic()

print('\n============================================')
print('              C2: Python Code')
print('============================================\n')
print('Checksum S = {:.6f}'.format(s))
print('Execution time = {:.6f} s\n'.format(end - start))

'''''''''''''''''''''''''''''
C3
'''''''''''''''''''''''''''''
x0 = np.array(x0)
w0 = np.array(w0)
w1 = np.array(w1)

start = monotonic()
z0 = x0.dot(w0).clip(0)
z1 = z0.dot(w1).clip(0)
s = z1.sum()
end = monotonic()

print('============================================')
print('              C3: NumPy Code')
print('============================================\n')
print('Checksum S = {:.6f}'.format(s))
print('Execution time = {:.6f} s\n'.format(end - start))





