#!/usr/bin/env python
'''
Uncoded matrix-vector multiplication
'''

from __future__ import division
from mpi4py import MPI
import numpy as np
import random
import threading
import time

# Change to True for more accurate timing, sacrificing performance
barrier = True

##################### Parameters ########################
#Conditions that need to be met for the parameters:
#kq|m
#k|n
#(k-1)|(m/N)
#m/N >> 1

#Use one master and N workers. Note that N should be k*q. 
#SOS Values are read from file if you load the matrices.
k = 10
q = 2
N = k*q

#Number of jobs i.e. number of matrix-vector multiplications c = A*b for different A, b. 
#SOS Values are read from file if you load the matrices.
J = q**(k-1)

#Set to 1 to load pregenerated list of A,b from current directory or 0 to generate them. Note that parameters of this script and of pregenerateAb_mat_vec.py should match.
#Also, matrices have to be in a folder "pregenerateAb_mat_vec" in parent directory.
loadAb = 0

#Bound of the elements of the input matrices i.e. those should be in [0,...,B-1].
#SOS Value is read from file if you load the matrices.
B = 3

#Input matrix size - A: m by n, b: n by 1. 
#SOS Values are read from file if you load the matrices.
m = 234000
n = 100

#Size of the matrices' entries in bits (CAUTION: it may be 32 or 64 for np.int_).
#For the AWS I tested np.int_ is 64 bits. You may change every occurrence of "np.int_" to "np.float32" but there may accuracy issues or other errors.
# T = 64
T = np.dtype(np.int_).itemsize*8

#########################################################

comm = MPI.COMM_WORLD

#If the matrices are loaded then the parameters are overwritten by those defined in "../pregenerateAb_mat_vec/config_lst". 
#Only the master node needs to have this file and then broadcast the parameters to the servers.
if loadAb == 1:
	
	if comm.rank == 0:
		config_lst = np.load('../pregenerateAb_mat_vec/config_lst.npz')
		k = int(config_lst['arr_0'])
		q = int(config_lst['arr_1'])
		B = int(config_lst['arr_2'])
		m = int(config_lst['arr_3'])
		n = int(config_lst['arr_4'])
	
#Master broadcasts and servers receive
k = comm.bcast(k, root=0)
q = comm.bcast(q, root=0)
B = comm.bcast(B, root=0)
m = comm.bcast(m, root=0)
n = comm.bcast(n, root=0)
N = k*q
J = q**(k-1)

#Check conditions
if comm.rank == 0:
	if comm.size != N+1:
		print("The number of MPI processes mismatches the number of workers.")
		comm.Abort(1)
	elif np.mod(m,N) != 0:
		print("N does not divide m.")
		comm.Abort(1)
	elif np.mod(n,k) != 0:
		print("k does not divide n.")
		comm.Abort(1)
	elif np.mod(m//N,k-1) != 0:
		print("k-1 does not divide m/N.")
		comm.Abort(1)

#Create a communicator only with servers. MPI convention: 0-indexed.
server_comm = comm.Split((comm.rank==0)*0 + (comm.rank>0)*1, comm.rank)

if comm.rank == 0:
	# Master
	print "Running with %d processes:" % comm.Get_size()

	print "UNCODED, N=%d workers, k=%d, q=%d, J=%d, m=%d, n=%d, B=%d" % (N, k, q, J, m, n, B)
	
	bp_start = MPI.Wtime()
	
	#test
	np.random.seed(1)
	
	#Create random matrices or load them from files
	A = []
	b = []
	if loadAb == 0:
		print('Generating matrices...')
		for i in range(J):
			A.append(np.asmatrix(np.random.randint(0,B,(m,n))).astype(np.int_)) #Convert np array to matrix
			b.append(np.asmatrix(np.random.randint(0,B,(n,1))).astype(np.int_)) #Convert np array to matrix
		
		#test		
		# np.savez('A_list', *A)
		# np.savez('b_list',*b)
		
	elif loadAb == 1:
		print('Loading matrices...')
		A_file = np.load('../pregenerateAb_mat_vec/A_list.npz')
		b_file = np.load('../pregenerateAb_mat_vec/b_list.npz')
		for i in range(J):
			A.append(A_file['arr_%d'%i])
			b.append(b_file['arr_%d'%i])
	
	#test
	# for i in range (J):
		# print('A[%d] is: ' % i)
		# print(A[i])
		# print('b[%d] is: ' % i)
		# print(b[i])
	
	#Split horizontally 
	Ah = [] 
	for i in range(J):
		Ah.append(np.split(A[i], q, axis=0))
		
	#Split vertically 
	Ahv = []
	bhv = []
	for i in range(J):
		Ahv_tmp = []
		for j in range(q):
			Ahv_tmp.append(np.split(Ah[i][j], k, axis=1))
			
		Ahv.append(Ahv_tmp)
		bhv.append(np.split(b[i], k, axis=0))
	
	#Initialize return dictionary
	Crtn = []
	for i in range(J):
		tmp_list = []
		
		#N reducers are used per job (one for each block row)
		for j in range(N):
			tmp_list.append(np.empty((m//N, 1), dtype=np.int_))
			
		Crtn.append(tmp_list)  

	#Start requests to send and receive
	reqc = [None] * J * N

	bp_end = MPI.Wtime()
	print "Pre-processing:            %f" % (bp_end - bp_start)
	
	bp_start = MPI.Wtime()

	#For each job
	for i in range(J):
	
		#test
		# bp_start_job = MPI.Wtime()
		
		for j in range(N):
			comm.Send(np.ascontiguousarray(Ahv[i][j//k][j%k]), dest=j+1, tag=15)
			Ahv[i][j//k][j%k] = None #Release the memory for the current server immediately since np.ascontiguousarray consumes more memory by itself
			comm.Send(np.ascontiguousarray(bhv[i][j%k]), dest=j+1, tag=29)
		bhv[i] = None #Release the memory for the current job immediately since np.ascontiguousarray consumes more memory by itself
		
		#test
		# bp_end_job = MPI.Wtime()
		# print "Job %d input transmission: %f" %(i+1, bp_end_job - bp_start_job)
		
		#N reducers are used per job (one for each block row). The +1 is due to the MPI-assigned ranks.
		for j in range(N):
			reqc[i*N+j] = comm.Irecv(Crtn[i][j], source=j+1, tag=42)

	#Optionally wait for all workers to receive their submatrices, for more accurate timing
	if barrier:
		comm.Barrier()

	bp_end = MPI.Wtime()
	print "Input transmission:        %f" %(bp_end - bp_start)
	
	print "The input has been transmitted. Limit the rate and press Enter to continue..."
	raw_input()
	
	#Master measures computation time
	comm.Barrier()
	bp_start_comp = MPI.Wtime()
	comm.Barrier()
	bp_end_comp = MPI.Wtime()
	print "Computation (MAP):         %f" %(bp_end_comp - bp_start_comp)
	
	#Master measures memory allocation time
	comm.Barrier()
	bp_end_mem = MPI.Wtime()
	print "Shuffle memory allocation: %f" %(bp_end_mem - bp_end_comp)
	
	#Master measures shuffling time
	comm.Barrier()
	bp_end_comm = MPI.Wtime()
	print "Communication (SHUFFLE):   %f" %(bp_end_comm - bp_end_mem)
	rate = (J*(k-1)*m*T)/(bp_end_comm - bp_end_mem)/1000**2
	print "Rate (Mbps):               %f" %(rate)
	
	#Master measures reduction time
	comm.Barrier()
	bp_end_red = MPI.Wtime()
	print "REDUCE:                    %f" %(bp_end_red - bp_end_comm)
	
	MPI.Request.Waitall(reqc)
	
	#test
	# print(type(Crtn[0]))
	
	#For each job, concatenate results of the reducers. Save returned product to file.
	c = []
	for i in range(J):
		cur_col = np.empty((0,1), dtype=np.int_)

		#construct column
		for j in range(N):
			cur_col = np.append(cur_col, Crtn[i][j], axis=0)
		
		#concatenate column
		c.append(cur_col)

	bp_start = MPI.Wtime()
	
	#Test
	# print("c_list", c)
	# np.savez('c_list', *c)
	
	bp_end = MPI.Wtime()
	print "Storing c to drive:        %f" %(bp_end - bp_start)
	print "Map + Shu + Red (no mem):  %f" %(bp_end_red - bp_start_comp - (bp_end_mem-bp_end_comp))
	print "Map + Shu + Red (mem):     %f" %(bp_end_red - bp_start_comp)
	print "T is:                      %d" %(T)


else:
	
	#For each job, create matrices (m/q)x(n/k) and (n/k)x(1)
	A = []
	b = []
	for i in range(J):
		A.append(np.empty((m//q,n//k), dtype=np.int_))
		# A.append(np.empty_like(np.matrix([[0]*(n//k) for j in range(m//q)])).astype(np.int_))
		b.append(np.empty((n//k,1), dtype=np.int_))
		# b.append(np.empty_like(np.matrix([[0]*(1) for j in range(n//k)])).astype(np.int_))
		comm.Recv(A[i], source=0, tag=15)
		comm.Recv(b[i], source=0, tag=29)

	#test
	# for i in range(J):
		# print "For job %d, worker %d received splits of A, b" % (i, comm.Get_rank()-1)
		# print(A[i])
		# print(b[i])
	
	if barrier:
		comm.Barrier()
	
	###################################################################################################################################################################################################
	#This barrier will be used for computation time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	c = []
	for i in range(J):
		# c.append(A[i]*b[i])
		c.append(np.matmul(A[i], b[i]))
	
	###################################################################################################################################################################################################
	#This barrier will be used for computation/memory allocation time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################

	#Initialize reduction dictionary for the number of jobs that I am a reducer
	Crtn = []
	for i in range(J):
		tmp_list = []
		
		#k-1 computations will be received for each reduction
		for j in range(k-1):
			tmp_list.append(np.empty((m//N, 1), dtype=np.int_))
			
		Crtn.append(tmp_list)  
	
	###################################################################################################################################################################################################
	#This barrier will be used for memory allocation/shuffling time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################

	#After the local computation, each reducer is receiving k-1 local computations from other workers (for each job)
	for i in range(J):
		
		rec_so_far = 0
		
		#For each block-row OF the q x k grid
		for j in range(q):
		
			#If I am in the current row
			if (comm.rank-1)//k == j:
			
				#For each transmitter
				for l in range(k):
				
					transmitter = j*k+l
					
					#The mappers that map the same block-row as me (0-indexed)
					receivers = [x for x in range((transmitter//k)*k, (transmitter//k)*k+k)]
					receivers.remove(transmitter)
					
					# print "Worker %d will send values to mappers %s" % (comm.Get_rank()-1, receivers)
					
					#test
					# if comm.rank == 4:
						# print(i,j,transmitter,receivers)
							
					if comm.rank-1 == transmitter:
					
						#I am the transmitter
						
						#For each receiver
						tx_req = [None] * (k-1)
						for x in range(k-1):
							receiver = receivers[x]
							# tx_req[x] = comm.Isend(c[i][(m//N)*(receiver%k):(m//N)*(receiver%k+1)], dest=receiver + 1, tag=5)
							comm.Send(c[i][(m//N)*(receiver%k):(m//N)*(receiver%k+1)], dest=receiver + 1, tag=5)
							
							#Debug
							# print np.shape(c[i][(m//N)*(receiver%k):(m//N)*(receiver%k+1)])[0]
							
							#The current transmitter needs to wait so that there is no parallel communication and the rate measurement should be accurate
							server_comm.Barrier()
							
						# MPI.Request.Waitall(tx_req)	
						
					else:
					
						#I am a receiver
						
						#Receive computation from the current transmitter of the block row
						for x in range(k-1):
							receiver = receivers[x]
							
							if comm.rank-1 == receiver:
								# rec_req = comm.Irecv(Crtn[i][rec_so_far], source=transmitter+1, tag=5)
								comm.Recv(Crtn[i][rec_so_far], source=transmitter+1, tag=5)
								
								# MPI.Request.Wait(rec_req)
								
								#Debug
								# print np.shape(Crtn[i][rec_so_far])[0]
								
								rec_so_far = rec_so_far + 1
			
							#The servers that participate in the communication for this row but are not the current transmitter need to wait so that there is no parallel communication and the rate measurement should be accurate
							server_comm.Barrier()
							
					#The servers that participate in the communication for this row but are not the current transmitter need to wait so that there is no parallel communication and the rate measurement should be accurate
					# server_comm.Barrier()
			
			#The servers that do not participate in the communication for this row need to wait so that there is no parallel communication and the rate measurement should be accurate
			server_comm.Barrier()
	
	#test
	# if comm.rank == 6:
		# print(Crtn)
		
	###################################################################################################################################################################################################
	#This barrier will be used for shuffling/reduction time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#We separate the Reduce operations so that we can do accurate timing of the communication
	reduced_blk = []
	for i in range(J):
		reduced_blk.append(np.sum(Crtn[i], axis=0) + c[i][(m//N)*((comm.rank-1)%k):(m//N)*((comm.rank-1)%k+1)])
		
		#test
		# if comm.rank == 4:
			# print(reduced_blk[i])
		
	###################################################################################################################################################################################################
	#This barrier will be used for reduction time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#Return all reductions to master for concatenation
	master_req = [None] * J
	for i in range(J):
		master_req[i] = comm.Isend(reduced_blk[i], dest=0, tag=42)
	MPI.Request.Waitall(master_req)
	
	#Return computation time to master
	# comm.send(comp_time, dest=0, tag=49)
