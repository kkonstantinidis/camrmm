# -*- coding: utf-8 -*-

import numpy as np
import sys

##################### Parameters ########################

# Number of jobs i.e. number of matrix-vector multiplications c = A*b for different A, b
k = 10
q = 2
J = q**(k-1)

#Bound of the elements of the input matrices i.e. those should be in [0,...,B]
B = 3

#Input matrix size - A: m by n, b: n by 1
m = 234000
n = 100

#Check parameters
if np.mod(m, k*q) != 0 or np.mod(n, k) != 0 or np.mod(m//(k*q), k-1) != 0:
	print("ERROR: The parameters are set wrong!")
	sys.exit()
	
#Save configuration so that you can load on execution
config_lst = [k, q, B, m, n]
np.savez('config_lst', *config_lst)
print("Configuration stored in file config_lst ")

# raw_input("The configuration has been saved. Copy it to UNCODED and CODED codes and press Enter to continue...")

print("Generating A (%dx%d), b(%dx1), B=%d for J=%d jobs" % (m, n, n, B, J))

#Generate and store matrices
A = [] 
b = []
for i in range(J):
	# A.append(np.asmatrix(np.random.randint(0,B,(m,n))).astype(np.int_))
	A.append(np.asmatrix(np.random.randint(0,B,(m,n))).astype(np.float32))
	# b.append(np.asmatrix(np.random.randint(0,B,(n,1))).astype(np.int_))
	b.append(np.asmatrix(np.random.randint(0,B,(n,1))).astype(np.float32))

print("A, b have been generated")

np.savez('A_list', *A)
np.savez('b_list', *b)

print("A, b have been stored")
