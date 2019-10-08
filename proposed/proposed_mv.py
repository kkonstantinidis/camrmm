#!/usr/bin/env python
'''
CAMR matrix-vector multiplication
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
#The code does not work if m/N < 2.
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

#Define the global variables that will be edited by the encoding function
Own = [[] for i in range(J)] #A 2D list where Own[i] are the owners of the i-th job. Convention: 1-indexed.
Own_batch = [[] for i in range(J)] #A 2D list where Own_batch[i][j] are the owners of the j-th batch of the i-th job. Convention: 1-indexed.
own_comm = [] #own_comm[i] is the communicator of the owners of the i-th job. MPI convention: 0-indexed.
mas_own_batch_comm = [[] for i in range(J)] #mas_own_batch_comm[i][j] is the communicator of the owners of the j-th batch of the i-th job. MPI convention: 0-indexed.

#A 2D list, where Blocks[i][j] will be populated by the block B_ij. Convention: 1-indexed.
Blocks = [[[] for i in range(q)] for j in range(k)]

groups_st2 = [] #A list where groups_st2[i] contains the COMM_WORLD ranks (MINUS ONE) of the i-th shuffling group of Stage 2. Convention: 0-indexed.
st2_comm = [] #st2_comm[i] is the communicator of the i-th group to communicate in Stage 2. MPI convention: 0-indexed.

#The following function generates all shuffling groups of Stage 2
#All servers need to call it with arguments (0, i, [], {}) for i = 0,1,...,q-1, i.e., for all possible choices for the first parallel class
#i: parallel class of current call
#j: server (block) of current parallel class
#cur_group: list of servers picked so far (one from each parallel class). Convention: 0-indexed.
#common_jobs: set of common jobs among all servers in cur_group. Convention: 1-indexed.
def generateStage2(i, j, cur_group, common_jobs):
	
	global Blocks
	global groups_st2
	global st2_comm
    
	#test
	# print(i,j,cur_group)
	
	if i < k-1:
	
		#Intermediate parallel class
		
		cur_group.append(i*q+j)
		
		if i == 0:
			common_jobs = set(Blocks[0][j])
		else:
			common_jobs = common_jobs.intersection(set(Blocks[i][j]))
		
		#Repeat for all blocks of the next parallel class
		for l in range(q):
			generateStage2(i+1, l, list(cur_group), common_jobs) #You need to call it with a copy of the list so that the recursion works correctly
			
	elif i == k-1:
	
		#Last parallel class
			
		#Find the unique last block that so that the intersection of the whole group is empty
		if len(common_jobs.intersection(set(Blocks[i][j]))) == 0:
			cur_group.append(i*q+j)
			groups_st2.append(cur_group)
			
			#Create the corresponding communicator for Shuffling Stage 2. Note that server_comm is 0-indexed like cur_group.
			if (server_comm.rank in cur_group):
				cur_st2_comm = server_comm.Split(1, server_comm.rank)
				st2_comm.append(cur_st2_comm)	
			else:
			
				#Unfortunately, there is no MPI_UNDEFINED in Python and since all machines need to call Split(), the non-owners pass color = 0 
				cur_st2_comm = server_comm.Split(0, server_comm.rank)
				st2_comm.append([])
			

def generateR():

	#A list of the parallel classes to be returned
	# ret = []

	global Blocks
	
	global Own
	
	#Generate the SPC code
	alphabet = range(q) #alphabet of the codewords
	T = np.zeros((k,1), dtype=np.int_) #codewords matrix
	cod = np.zeros((1,k), dtype=np.int_) #a single codeword
	for i in range(1,J):
		cod = cod + np.concatenate((np.zeros((1,k-2)), np.expand_dims(np.array([1,0]), axis=0)), axis=1).astype(np.int_)
		for j in range(k-2,0,-1):
			if cod[0][j] == q:
				cod[0][j] = 0
				cod[0][j-1] = cod[0][j-1] + 1
			else:
				break
		cod[0][k-1] = np.mod(np.sum(cod[0][0:k-1]), q)
		T = np.concatenate((T, np.transpose(cod)), axis=1)
		
	#Construct SPC blocks i.e. the jobs that each server owns. Also populate the list of owners for each job.
	for i in range(k):
		for j in range(q):
			Blocks[i][j] = np.flatnonzero(T[i,:] == j)+1
			for l in range(len(Blocks[i][j])):
				Own[Blocks[i][j][l]-1].append(i*q+j+1)
		
	#Create a communicator for each job and include the master and the owners in that
	# mas_own_comm = []
	global own_comm
	global mas_own_batch_comm
	for i in range(J):
		# mas_own_comm.append(comm.Split(((comm.rank==0) + (comm.rank in Own[i]))*1, comm.rank))
		
		#Get the communicator's group and create a new communicator after keeping only the server-owners
		# mas_own_group = mas_own_comm[i].Get_group()
		# own_group = mas_own_group.Incl(k, Own[i])
		# own_comm.append(mas_own_comm.Create_group(own_group))
		
		#test
		# if i == 2:
			# print(Own[i])
			
		#Create a new communicator for each job after keeping only its server-owners
		#For some reason, ranks are not assigned 0-based in the new communicator
		#Non-owners of the job will not store this communicator
		if (comm.rank in Own[i]):
			cur_own_comm = comm.Split(1, comm.rank)
			own_comm.append(cur_own_comm)	
		else:
		
			#Unfortunately, there is no MPI_UNDEFINED in Python and since all machines need to call Split(), the non-owners pass color = 0 
			cur_own_comm = comm.Split(0, comm.rank)
			own_comm.append([])
		
		# own_comm.append(comm.Split((comm.rank not in Own[i])*0 + (comm.rank in Own[i])*1, comm.rank))
		
		#test
		# if i == 3:
			# if (comm.rank in Own[i]):
				# print "MPI node %d is owner of job %d and within that has rank %d" % (comm.Get_rank(), i+1, own_comm[i].rank)
			# else: 
				# print "MPI node %d is NOT owner of job %d" % (comm.Get_rank(), i+1)
		
		for j in range(k):
		
			#Create a new communicator for each batch after keeping only the batch owners
			batch_own_ind = [l for l in range(k) if l != j]
			batch_own_ranks = [Own[i][l] for l in batch_own_ind] 
			Own_batch[i].append(batch_own_ranks)
			
			#test
			# if (comm.rank == 1 and (comm.rank in Own_batch[i][j])):
				# print(Own_batch[i][j])
				
			#test
			# if i == 0 and j == 0:
				# print(batch_own_ranks)
			
			#For some reason, ranks are not assigned 0-based in the new communicator
			#Non-owners of the batch, except the master will not store this communicator
			if (comm.rank == 0 or (comm.rank in batch_own_ranks)):
				cur_mas_own_batch_comm = comm.Split(1, comm.rank)
				mas_own_batch_comm[i].append(cur_mas_own_batch_comm)
			else:
				cur_mas_own_batch_comm = comm.Split(0, comm.rank)
				mas_own_batch_comm[i].append([])
				
			#test
			# if i == 3:
				# if (comm.rank == 0 or (comm.rank in batch_own_ranks)):
					# print "MPI node %d is master OR owner of batch %d of job %d and within that has rank %d" % (comm.Get_rank(), j+1, i+1, mas_own_batch_comm[i][j].rank)
				# else: 
					# print "MPI node %d is NOT owner of batch %d of job %d" % (comm.Get_rank(), j+1, i+1)
				
			# mas_own_batch_comm[i].append(comm.Split(((comm.rank==0) + (comm.rank in batch_own_ranks))*1, comm.rank))
			
			# batch_own_group = own_group.Incl(k-1, batch_own_rank)
			# own_batch_comm[i].append(own_comm[i].Create_group(batch_own_group))
			

if comm.rank == 0:
	print "Running with %d processes:" % comm.Get_size()
	
	print "CAMR, N=%d workers, k=%d, q=%d, J=%d, m=%d, n=%d, B=%d" % (N, k, q, J, m, n, B)
	
	print "Generating SPC code..."

#START of code generation
comm.Barrier()
bp_start_codegen = MPI.Wtime()

#Generate SPC code
generateR()

if comm.rank == 0:
	print "Generating Stage 2 groups..."
	
#Only the servers will generate the shuffling groups of Stage 2
if comm.rank > 0:
	for i in range(q):
		generateStage2(0, i, [], {})

#END of code generation
#Code generation is necessary for the remaining operations, but this barrier may not be needed
comm.Barrier()
bp_end_codegen = MPI.Wtime()
	
if comm.rank == 0:

	# Master
	
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
		
	#Split A only vertically, b is stored as row vector, but still split it into k parts
	Av = []
	bv = []
	for i in range(J):
		Av.append(np.split(A[i], k, axis=1))
		bv.append(np.split(b[i], k, axis=0))
	
	#Initialize return dictionary
	Crtn = []
	for i in range(J):
		tmp_list = []
		
		#N reducers are used per job (one for each block row)
		for j in range(N):
			tmp_list.append(np.empty((m//N, 1), dtype=np.int_))
			
		Crtn.append(tmp_list)  

	#Sends are synchronous but for the reception, the reducers are all N servers
	reqc = [None] * J * N
	
	bp_end = MPI.Wtime()
	print "Pre-processing:                                 %f" % (bp_end - bp_start)

	bp_start = MPI.Wtime()

	#For each job
	for i in range(J):
	
		#test
		# bp_start_job = MPI.Wtime()
		
		#For each file batch
		for j in range(k):
			mas_own_batch_comm[i][j].Bcast(np.ascontiguousarray(Av[i][j]), root=0)
			mas_own_batch_comm[i][j].Bcast(np.ascontiguousarray(bv[i][j]), root=0)
			
			#test
			# print("Master transmitting A: ", Av[i][j])
			# print("Master transmitting b: ", bv[i][j])
			
			#Release the memory for the current transmission immediately since np.ascontiguousarray consumes more memory by itself
			Av[i][j] = None
			bv[i][j] = None
			
			#test
			dummy = 1
		
		#test
		# bp_end_job = MPI.Wtime()
		# print "Job %d input transmission: %f" %(i+1, bp_end_job - bp_start_job)
		
		#N reducers are used per job (one for each block row). Indexing starts from zero. The +1 is due to the MPI-assigned ranks.
		for j in range(N):
			reqc[i*N+j] = comm.Irecv(Crtn[i][j], source=j+1, tag=20)

	# Optionally wait for all workers to receive their submatrices, for more accurate timing
	if barrier:
		comm.Barrier()
		
	bp_end = MPI.Wtime()
	print "Input transmission:                             %f" %(bp_end - bp_start)
	
	print "The input has been transmitted. Limit the rate and press Enter to continue..."
	raw_input()
	
	print "CodeGen:                                        %f" %(bp_end_codegen - bp_start_codegen)
	
	#Master measures computation time
	comm.Barrier()
	bp_start_comp = MPI.Wtime()
	comm.Barrier()
	bp_end_comp = MPI.Wtime()
	print "Computation (MAP):                              %f" %(bp_end_comp - bp_start_comp)
	
	#Master measures encoding time
	comm.Barrier()
	bp_end_enc = MPI.Wtime()
	print "Encoding:                                       %f" %(bp_end_enc - bp_end_comp)
	
	#Master measures memory allocation time
	comm.Barrier()
	bp_end_mem = MPI.Wtime()
	print "Shuffle memory allocation:                      %f" %(bp_end_mem - bp_end_enc)
	
	#Master measures shuffling time
	#Receive total shuffling time and transmission rate from workers
	sh_time = 0
	tx_rate = 0
	for i in range(N):
		sh_time = sh_time + comm.recv(source=i+1, tag=55)
		tx_rate = tx_rate + comm.recv(source=i+1, tag=56)
		
	comm.Barrier()
	bp_end_comm = MPI.Wtime()
	print "Communication (SHUFFLE):                        %f" %(sh_time)
	
	#The rate is computed by considering the amount of data of broadcasts as if they were implemented as unicasts
	# stage1_load = J*k/(k-1)*m/N*T
	# stage2_load = q**(k-1)*(q-1)*k/(k-1)*m/N*T
	# stage3_load = N*(J-q**(k-2))*m/N*T
	# rate = ((k-1)*stage1_load+(k-1)*stage2_load+stage3_load)/(bp_end_comm - bp_end_mem)/1000**2
	print "Rate (Mbps):                                    %f" %(tx_rate/N)
	
	#Master measures decoding time
	comm.Barrier()
	bp_end_dec = MPI.Wtime()
	print "Decoding:                                       %f" %(bp_end_dec - bp_end_comm)
	
	#Master measures reduction time
	comm.Barrier()
	bp_end_red = MPI.Wtime()
	print "REDUCE:                                         %f" %(bp_end_red - bp_end_dec)
	
	MPI.Request.Waitall(reqc)
	          
	#For each job, concatenate results of the reducers. Save returned product to file.
	c_list = []
	for i in range(J):
		cur_col = np.empty((0,1), dtype=np.int_)

		#construct column
		for j in range(N):
			cur_col = np.append(cur_col, Crtn[i][j], axis=0)
		
		#concatenate column
		c_list.append(cur_col)

	bp_start = MPI.Wtime()
	
	#Test
	# print("c_list", c_list)
	# np.savez('c_list', *c_list)
	
	bp_end = MPI.Wtime()
	print "Storing c to drive:                             %f" %(bp_end - bp_start)
	print "CodeGen + Map + Enc + Shu + Dec + Red (no mem): %f" %((bp_end_codegen - bp_start_codegen) + (bp_end_enc - bp_start_comp) + sh_time + (bp_end_red - bp_end_comm))
	print "CodeGen + Map + Enc + Shu + Dec + Red (mem):    %f" %((bp_end_codegen - bp_start_codegen) + (bp_end_mem - bp_start_comp) + sh_time + (bp_end_red - bp_end_comm))
	print "T is:                                           %d" %(T)
	
else:
	
	#For each job, the server will have a list of matrices (m)x(n/k) and (n/k)x(1).
	A = [[] for i in range(J)]
	b = [[] for i in range(J)]
	
	#For each job
	for i in range(J):
	
		#For each file batch
		batches_stored = 0
		for j in range(k):
			
			if (comm.rank in Own_batch[i][j]):		
				A[i].append(np.empty((m,n//k), dtype=np.int_))
				b[i].append(np.empty((n//k,1), dtype=np.int_))
				mas_own_batch_comm[i][j].Bcast(A[i][batches_stored], root=0)
				mas_own_batch_comm[i][j].Bcast(b[i][batches_stored], root=0)
				batches_stored = batches_stored + 1
				
				#test
				# if i == 0 and j == 1:
					# print "Worker %d is owner/received batch %d of job %d and within that has rank %d" % (comm.Get_rank(), j+1, i+1, mas_own_batch_comm[i][j].rank)
				
		
	#test
	# if comm.rank == 6:
		# for i in range(J):
			# if comm.rank in Own[i]:
				# print "For job %d, worker %d received splits of A, b" % (i+1, comm.Get_rank())
				# print(A[i])
				# print(b[i])
	
	if barrier:
		comm.Barrier()
	
	###################################################################################################################################################################################################
	#This barrier will be used for computation time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#Start the computation
	c = [[] for i in range(J)]
	for i in range(J):
	
		#If I am owner of this job
		if (comm.rank in Own[i]):
			
			batches_comp = 0 #number of batches computed so far
			for j in range(k):

				#If I am owner of this batch
				if (comm.rank in Own_batch[i][j]):
				
					#test
					# print i+1, j+1
					
					c[i].append(np.matmul(A[i][batches_comp], b[i][batches_comp]))
						
					#test
					# if comm.rank == 1:
						# print "For job %d, worker %d computed c" % (i+1, comm.Get_rank())
						# print(c[i][batches_comp])
						
					batches_comp = batches_comp + 1
	
	###################################################################################################################################################################################################
	#This barrier will be used for computation/encoding time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#ENCODING FOR STAGE 1
	#We need the encoded packets to be able to split into k-1 parts before transmission and since each of them is a vector we can attach some dummy zeros
	packet_size = (m//N)//(k-1) #needs to be global
		
	#test
	# if comm.rank == 1:
		# print "The decided packet size is: ", packet_size
		
	c_enc_st1 = [[] for i in range(J)]
	
	#Start the encoding
	for i in range(J):

		#If I am owner of this job
		if (comm.rank in Own[i]):
		
			#test
			# print "I own job ", i+1
			
			c_enc_st1[i] = np.zeros((packet_size, 1)).astype(np.int_) #to save encoded packet for current job
			batches_enc = 0 #number of batches used for encoding of this job so far
			for j in range(k):
			
				#If I am owner of this batch
				if (comm.rank in Own_batch[i][j]):
				
					#test
					# print "I own batch (i,j) = (", i+1, ", ", j+1, ")"
		
					#Find the unique owner of the job that does not own the current batch
					batch_non_owner = (set(Own[i]).difference(set(Own_batch[i][j]))).pop()
					
					#test
					# print "I own batch (i,j) = (", i+1, ", ", j+1, ") whose non owner is ", batch_non_owner
					
					#Pick only the rows of the batch's non-owner for encoding purposes
					c_uncoded = c[i][batches_enc][(m//N)*(batch_non_owner-1):(m//N)*batch_non_owner]
					
					#test
					# print "c_uncoded", c_uncoded
										
					#Figure out the index of the chunk I will be transmitting based on the rank
					tx_ind = Own_batch[i][j].index(comm.rank)
					
					#test
					# print "my tx_index is: ", tx_ind
					
					#Encode using only the rows (i.e. the chunk) that I will be transmitting
					c_enc_st1[i] = c_enc_st1[i] + c_uncoded[packet_size*tx_ind:packet_size*(tx_ind+1)]
					
					#test
					# print "my c_enc_st1 is: ", c_enc_st1
					
					batches_enc = batches_enc + 1
					
	
	#ENCODING FOR STAGE 2
	#Packet size is the same as in Stage 1
		
	c_enc_st2 = [[] for i in range(len(groups_st2))]
	
	#Start the encoding
	for i in range(len(groups_st2)):

		cur_group = groups_st2[i]
		
		#test
		print_group = [l+1 for l in cur_group]
		
		#If I am in current shuffling group
		if (server_comm.rank in cur_group):
		
			#test
			# if comm.rank == 4:
				# print "I am server ", comm.rank, " in Stage 2 group ", i+1, " consisting of ", cur_group
			
			c_enc_st2[i] = np.zeros((packet_size, 1), dtype=np.int_) #to save encoded packet for current group
			batches_enc = 0 #number of batches used for encoding of this packet so far
			
			#Remove one server at a time, except myself, from group and determine the (unique) common job & batch that the remaining servers (including myself) share
			for j in range(k):
			
				if j != st2_comm[i].rank:
					cur_sub_group = [l+1 for l in cur_group if l != cur_group[j]] #Note: my own rank is in cur_sub_group, and cur_sub_group is 1-indexed.
					
					#Find our common job (1-indexed)
					common_job = set(Blocks[(cur_sub_group[0]-1)//q][(cur_sub_group[0]-1)%q])
					for l in range(1, len(cur_sub_group)):
						common_job = common_job.intersection(set(Blocks[(cur_sub_group[l]-1)//q][(cur_sub_group[l]-1)%q]))
					
					#Convert set of 1 element to int by popping
					common_job = common_job.pop()
						
					#Also find our common batch (0-indexed) (there has to be one by construction)
					common_batch = None
					for b in range(k):
						if set(cur_sub_group).issubset(set(Own_batch[common_job-1][b])):
							common_batch = b
							break
							
					#test
					#Everything in the following print is 1-indexed
					# if comm.rank == 1:
						# print "For stage 2 group ", print_group, " and subgroup ", cur_sub_group, " we share job ", common_job, " and batch ", common_batch+1
					
					#The unique non-owner of the current batch is the excluded server of the group (0-indexed)
					batch_non_owner = cur_group[j]
					
					#Pick only the rows of the batch's non-owner for encoding purposes
					c_uncoded = c[common_job-1][batches_enc][(m//N)*batch_non_owner:(m//N)*(batch_non_owner+1)]
					
					#Figure out the index of the chunk I will be transmitting based on the rank
					tx_ind = Own_batch[common_job-1][common_batch].index(comm.rank)
				
					#Encode using only the rows (i.e. the chunk) that I will be transmitting
					c_enc_st2[i] = c_enc_st2[i] + c_uncoded[packet_size*tx_ind:packet_size*(tx_ind+1)]
					
					batches_enc = batches_enc + 1
					
			#test
			# if comm.rank == 4:
				# print "Server ", comm.rank, ": For stage 2 group ", print_group, " c_enc_st2 is ", c_enc_st2[i]

	
	###################################################################################################################################################################################################
	#This barrier will be used for encoding/memory allocation time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#Initialize receive buffers for Shuffle stage 1
	c_rec_st1 = [[] for i in range(J)]
	for i in range(J):
		
		#If I am owner of this job
		if (comm.rank in Own[i]):
			for j in range(len(Own[i])-1):
				c_rec_st1[i].append(np.empty((packet_size,1), dtype=np.int_))
			
	#Initialize receive buffers for Shuffle stage 2
	c_rec_st2 = [[] for i in range(len(groups_st2))] #Note that elements is as many as the number of groups (for decoding)
	for i in range(len(groups_st2)):
		
		#If I am in current shuffling group
		if (server_comm.rank in groups_st2[i]):
			for j in range(len(groups_st2[i])-1):
				c_rec_st2[i].append(np.empty((packet_size,1), dtype=np.int_))
	
	#Initialize receive buffers for Shuffle stage 3
	c_rec_st3 = [[] for i in range(J)] #Note that elements is as many as the number of jobs (for reduction)
	for i in range(J):
		
		#If I am NOT owner of this job
		if (comm.rank not in Own[i]):
			c_rec_st3[i].append(np.empty((m//N,1), dtype=np.int_))
			
	###################################################################################################################################################################################################
	#This barrier will be used for memory allocation/shuffling time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#Start the shuffling
	#For each worker, we will finish all the transmissions he has to do, so that we can accurately measure the rate
	packets_rec_st1 = [0 for x in range(J)]
	packets_rec_st2 = [0 for x in range(len(groups_st2))]
	for activeId in range(N):
		server_comm.Barrier()
		if (comm.rank == activeId+1):
			rTime = MPI.Wtime()
			txTime = 0
			tolSize = 0
		
		#SHUFFLE STAGE 1 - OWNERS OF EACH JOB COMMUNICATE
		for job in range(len(Blocks[activeId//q][activeId%q])):
		
			curr_job = Blocks[activeId//q][activeId%q][job]
			
			#If I or the transmitter are not owners of this job
			if (comm.rank not in Own[curr_job-1] or activeId+1 not in Own[curr_job-1]):
				continue
			
			mcComm = own_comm[curr_job-1]
			
			if (comm.rank == activeId+1):
			
				#I am transmitting
				
				#test
				# if i == 3:
					# print "For job ", curr_job, ", I am transmitter ", comm.rank, " (local rank = ", mcComm.rank, ") and I will send", c_enc_st1[curr_job-1], "of size ", packet_size
					
				bp_start = MPI.Wtime()
				mcComm.Bcast(c_enc_st1[curr_job-1], root=mcComm.rank) #indexing of sub-communicator starts from 0
				bp_end = MPI.Wtime()
				txTime = txTime + (bp_end-bp_start)
				tolSize = tolSize + packet_size
				
				#Debug
				# print np.shape(c_enc_st1[curr_job-1])[0]
				
			else:
			
				#I am receiving
				
				#Convert activeId to rootId of a particular multicast group
				for rootId in range(len(Own[curr_job-1])):
					if (Own[curr_job-1][rootId] == activeId+1):
						break
			
				#test
				# if i == 3:
					# print "For job ", curr_job, ", I am receiver ", comm.rank, " (local rank = ", mcComm.rank, ") and will receive from local rank ", rootId
			
				mcComm.Bcast(c_rec_st1[curr_job-1][packets_rec_st1[curr_job-1]], root=rootId) #indexing of sub-communicator starts from 0
				packets_rec_st1[curr_job-1] = packets_rec_st1[curr_job-1] + 1
			
			
		#SHUFFLE STAGE 2 - OWNERS AND NON-OWNERS COMMUNICATE
		for group in range(len(groups_st2)):
		
			#If I or the transmitter are not in current shuffling group
			if (server_comm.rank not in groups_st2[group] or activeId not in groups_st2[group]):
				continue
			
			#test
			# if comm.rank == 1:
				# print("Server ", comm.rank, " in stage 2 group ", group)
					
			mcComm = st2_comm[group]
			
			if (comm.rank == activeId+1):
			
				#I am transmitting
				
				bp_start = MPI.Wtime()
				mcComm.Bcast(c_enc_st2[group], root=mcComm.rank) #indexing of sub-communicator starts from 0
				bp_end = MPI.Wtime()
				txTime = txTime + (bp_end-bp_start)
				tolSize = tolSize + packet_size
				
				#Debug
				# print np.shape(c_enc_st2[group])[0]
				
			else:
			
				#I am receiving
				
				#Convert activeId to rootId of a particular multicast group
				for rootId in range(len(groups_st2[group])):
					if (groups_st2[group][rootId] == activeId):
						break

				#test
				# if comm.rank == 1:
					# print(len(c_rec_st2[group]))
					# print(packets_rec_st2[group])
				
				mcComm.Bcast(c_rec_st2[group][packets_rec_st2[group]], root=rootId) #indexing of sub-communicator starts from 0
				packets_rec_st2[group] = packets_rec_st2[group] + 1		
			
			
		#SHUFFLE STAGE 3 - OWNERS AND NON-OWNERS COMMUNICATE	
		#test
		# print "Stage 3 I am rank ", server_comm.rank
		
		#Find transmitter's parallel class
		tx_class = activeId//q
		
		#If I belong to this parallel class
		if server_comm.rank//q == tx_class:
		
			#Servers in current parallel class (0-indexed).
			p_class = set(range(tx_class*q, tx_class*q+q))
			
			#test
			# if tx_class == 0:
				# print(p_class)
			
			#The remaining servers are all receivers
			receivers = p_class.difference({activeId})

			for l in range(q-1): #For each receiver (we don't do broadcasts but unicasts)
				
				#pop() function is deterministic for sets of numbers, i.e., it always returns the first element
				receiver = receivers.pop()

				if server_comm.rank == activeId:
				
					#I am transmitting
					
					#test
					# print "Stage 3: I am TRANSMITTER with rank ", activeId+1, " and transmitting to ", receiver+1, " in parallel class ", tx_class+1
					
					for a in range(q**(k-2)): #For each job I own (receiver does not own any of these jobs by being in the same parallel class)
						
						#Aggregate all batches for current job and the columns of the receiver
						c_agg = c[Blocks[activeId//q][activeId%q][a]-1][0][(m//N)*receiver:(m//N)*(receiver+1)]
						for b in range(1, k-1):
							c_agg = c_agg + c[Blocks[activeId//q][activeId%q][a]-1][b][(m//N)*receiver:(m//N)*(receiver+1)]
							
						#test
						# if comm.rank == 2:
							# print "Stage 3: I am TRANSMITTER with rank ", activeId+1, " and transmitting ", c_agg, " to ", receiver+1, " for job ", Blocks[activeId//q][activeId%q][b]
						
						#test
						# if comm.rank == 2:
							# print("Stage 3 sent: ", Blocks[activeId//q][activeId%q][a])
							
						bp_start = MPI.Wtime()
						server_comm.Send(c_agg, dest=receiver, tag=10)
						bp_end = MPI.Wtime()
						txTime = txTime + (bp_end-bp_start)
						tolSize = tolSize + m//N
					
						#Debug
						# print np.shape(c_agg)[0]
					
				if server_comm.rank == receiver:
				
					#I am receiving
	
					for a in range(q**(k-2)):
					
						#test
						# if comm.rank == 2:
							# print(len(c_rec_st3))
							# print(Blocks[activeId//q][activeId%q][a]-1)
							
						#Each iteration refers to a different job that I do not own
						server_comm.Recv(c_rec_st3[Blocks[activeId//q][activeId%q][a]-1][0], source=activeId, tag=10)
						
						#test
						# if comm.rank == 6:
							# print("Stage 3 received: ", Blocks[activeId//q][activeId%q][a])
							# print("Stage 3 received: ", c_rec_st3[Blocks[activeId//q][activeId%q][a]-1])	
			
		server_comm.Barrier()
		if (comm.rank == activeId+1):
			rTime = MPI.Wtime() - rTime

	txRate = (tolSize*T)/txTime/1000**2 #in Mbps
	
	#Return total shuffling time and transmission rate in Mbps to master
	comm.send(rTime, dest=0, tag=55)
	comm.send(txRate, dest=0, tag=56)
	
	###################################################################################################################################################################################################
	#This barrier will be used for communication/decoding time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	#DECODING AND REDUCTION STARTS BELOW
	
	c_reduced = [np.asmatrix(np.empty((m//N,1), dtype=np.int_)) for i in range(J)]
	
	#test
	# print "c_reduced[0] type: ", type(c_reduced[0])
	
	#DECODING & REDUCTION FOR STAGE 1
	for i in range(J):
	
		#If I am owner of this job
		if (comm.rank in Own[i]):
		
			packets_dec = 0
			for j in range(len(Own[i])):
			
				transmitter = j
				
				#What is the transmitter's rank?
				transmitter_rank = Own[i][j]
				
				#I need to decode something only if I am a receiver
				if comm.rank != transmitter_rank:
					
					#Decoded packet will be stored here
					dec_packet = c_rec_st1[i][packets_dec]
					
					#test
					# if i == 0 and comm.rank == 1:
						# print "I am server ", comm.rank, " and for my owned job ", i+1, " I had received ", dec_packet, " from server ", transmitter_rank
				
					#Find all batches that I share with the transmitter (0-indexed) (there have to be k-1 by construction) and for each of them figure out who doesn't have it and decode
					for b in range(k):
						if set({comm.rank, transmitter_rank}).issubset(set(Own_batch[i][b])):
						
							#Find the unique owner of the job that does not own the current batch
							batch_non_owner = (set(Own[i]).difference(set(Own_batch[i][b]))).pop()
							
							#test
							# if i == 0 and comm.rank == 1 and transmitter_rank == 3:
								# print "I am server ", comm.rank, " and for my owned job ", i+1, " I share batch ", b+1, " with transmitting server ", transmitter_rank, " whose non-owner is ", batch_non_owner
							
							#Pick the computation for the corresponding batch that we share (note: servers compute batches in order 1,2,...,k except the batch B_{own_comm[i].rank}).
							#Pick only the rows of the batch's non-owner for decoding purposes
							if b > own_comm[i].rank:
								c_uncoded = c[i][b-1][(m//N)*(batch_non_owner-1):(m//N)*batch_non_owner]
							else:
								c_uncoded = c[i][b][(m//N)*(batch_non_owner-1):(m//N)*batch_non_owner]
							
							#test
							# if i == 0 and comm.rank == 1 and transmitter_rank == 3:
								# print "I am server ", comm.rank, " and for my owned job ", i+1, " I computed batch ", b+1, " with value ", c_uncoded
								
							#Figure out the index of the chunk that the transmitter has sent based on his rank
							tx_ind = Own_batch[i][b].index(transmitter_rank)
							
							#test
							# if np.shape(dec_packet)[0] == 0:
								# print "Job: ", i+1, ", transmitter: ", transmitter_rank, "shape: ", np.shape(dec_packet)
								# print "Job: ", i+1, ", transmitter: ", transmitter_rank, "shape: ", np.shape(c_rec_st1[i][packets_dec])
							
							#Decode using only the rows (i.e. the chunk) that has been received		
							dec_packet = dec_packet - c_uncoded[packet_size*tx_ind:packet_size*(tx_ind+1)]
							
							#test
							# if i == 0 and comm.rank == 1 and transmitter_rank == 3:
								# print "I am server ", comm.rank, " and for my owned job ", i+1, " I computed batch ", b+1, " with dec_packet ", dec_packet
							

					#test
					# if np.shape(dec_packet)[0] == 0:
						# print "Job: ", i+1, ", transmitter: ", transmitter_rank, "shape: ", np.shape(dec_packet)
					
					#By convention, I am missing the batch indexed with my rank
					my_missing_batch = own_comm[i].rank
					
					#Who are its owners?
					mis_bat_owners = Own_batch[i][my_missing_batch]
					
					c_reduced[i][(m//N)//(k-1)*mis_bat_owners.index(transmitter_rank):(m//N)//(k-1)*(mis_bat_owners.index(transmitter_rank)+1)] = dec_packet
					
					packets_dec = packets_dec + 1
					
			#Add my local computation
			for l in range(len(c[i])):
				c_reduced[i] = c_reduced[i]+ c[i][l][(m//N)*server_comm.rank:(m//N)*(server_comm.rank + 1)]
				
				#test
				# if i == 0 and comm.rank == 1:
					# print "c[i][l] = ", c[i][l]
					# print "c[i][l] type: ", type(c[i][l])
			
			#test
			# if i == 2 and comm.rank == 6:
				# print "I am server ", comm.rank, " and for my owned job ", i+1, " I have reduced ", c_reduced[i]
				
				
	#DECODING & REDUCTION FOR STAGE 2
	for i in range(len(groups_st2)):
	
		cur_group = groups_st2[i]
		
		#If I am in current shuffling group
		if (server_comm.rank in groups_st2[i]):
		
			packets_dec = 0
			
			#Figure out the COMM_WORLD MINUS ONE ranks of the remaining group (0-indexed)
			rest_group = list(cur_group)
			rest_group.remove(server_comm.rank)
			
			#Since, I am not owner of the job I am decoding I need to figure out what it is, i.e., what is the job the remaining servers in current group share (1-indexed)
			rest_com_job = set(Blocks[(rest_group[0])//q][(rest_group[0])%q])
			for l in range(1, len(rest_group)):
				rest_com_job = rest_com_job.intersection(set(Blocks[(rest_group[l])//q][(rest_group[l])%q]))
			
			#Convert set of 1 element to int by popping
			rest_com_job = rest_com_job.pop()
			
			#test
			# if comm.rank == 6:
				# print "In stage 2, I (server ", comm.rank-1, ") was in group ", cur_group, " and a remaining group is ", rest_group, " with common job ", rest_com_job
				
			#For each possible transmitter, decode a packet
			for j in range(len(groups_st2[i])):
			
				transmitter = j
				
				#What is the transmitter's rank?
				transmitter_rank = groups_st2[i][j] + 1
				
				#I need to decode something only if I am a receiver
				if st2_comm[i].rank != transmitter:
				
					#Decoded packet will be stored here
					dec_packet = c_rec_st2[i][packets_dec]
					
					#test
					# if i == 1 and comm.rank == 1:
						# print "I am server ", comm.rank, " and for stage 2 group ", i+1, " I had received ", dec_packet, " from server ", transmitter_rank
					
					#Remove one server at a time, except myself and the transmitter, from group and determine the (unique) common job & batch that the remaining servers (including myself) share
					for x in range(len(groups_st2[i])):
						
						if x != st2_comm[i].rank and x != transmitter:
							cur_sub_group = [l+1 for l in cur_group if l != cur_group[x]] #Note: my own rank is in cur_sub_group, and cur_sub_group is 1-indexed.
							
							#Find our common job (1-indexed)
							common_job = set(Blocks[(cur_sub_group[0]-1)//q][(cur_sub_group[0]-1)%q])
							for l in range(1, len(cur_sub_group)):
								common_job = common_job.intersection(set(Blocks[(cur_sub_group[l]-1)//q][(cur_sub_group[l]-1)%q]))
							
							#Convert set of 1 element to int by popping
							common_job = common_job.pop()
								
							#Also find our common batch (0-indexed) (there has to be one by construction)
							common_batch = None
							for b in range(k):
								if set(cur_sub_group).issubset(set(Own_batch[common_job-1][b])):
									common_batch = b
									break
							
							#The unique non-owner of the current batch is the excluded server of the group (0-indexed)
							batch_non_owner = cur_group[x]
							
							#test
							# if i == 3 and comm.rank == 6:
								# print "Transmitter is ", transmitter_rank
								# print "I am server ", comm.rank, " and for stage 2 group ", i+1, " and subgroup ", cur_sub_group, " we share job ", common_job, " and batch ", common_batch+1, " whose non-owner is ", batch_non_owner+1
						
							#Pick the computation for the corresponding batch that we share (note: servers compute batches in order 1,2,...,k except the batch B_{own_comm[i].rank}).
							#Pick only the rows of the batch's non-owner for decoding purposes
							if common_batch > own_comm[common_job-1].rank:
								c_uncoded = c[common_job-1][common_batch-1][(m//N)*batch_non_owner:(m//N)*(batch_non_owner+1)]
							else:
								c_uncoded = c[common_job-1][common_batch][(m//N)*batch_non_owner:(m//N)*(batch_non_owner+1)]
							
							#Figure out the index of the chunk that the transmitter has sent based on his rank
							tx_ind = Own_batch[common_job-1][common_batch].index(transmitter_rank)
							
							#Decode using only the rows (i.e. the chunk) that has been received		
							dec_packet = dec_packet - c_uncoded[packet_size*tx_ind:packet_size*(tx_ind+1)]
	
					#The remaining servers including the transmitter share a common job and a common batch for that job.
					c_reduced[rest_com_job-1][(m//N)//(k-1)*rest_group.index(transmitter_rank-1):(m//N)//(k-1)*(rest_group.index(transmitter_rank-1)+1)] = dec_packet
					
					packets_dec = packets_dec + 1
				
				
			#test
			# if i+1 == 4 and comm.rank == 6:
				# print "I am server ", comm.rank, " and for stage 2 group ", i+1, " I have reduced ", c_reduced[rest_com_job-1]
				
	
	###################################################################################################################################################################################################
	#This barrier will be used for decoding/reduction time measurement by the master
	comm.Barrier()
	###################################################################################################################################################################################################
	
	
	#REDUCTION FOR STAGE 3
	for i in range(k): #For each parallel class
	
		#If I belong to this parallel class
		if server_comm.rank//q == i:
			
			#Servers in current parallel class (0-indexed).
			p_class = set(range(i*q,i*q+q))

			for j in range(q): #For each transmitter
				
				transmitter = i*q+j
				
				#The remaining servers were all receivers
				receivers = p_class.difference({transmitter})
				
				for l in range(q-1): #For each receiver (we didn't do broadcasts but unicasts)
				
					#pop() function is deterministic for sets of numbers, i.e., it always returns the first element
					receiver = receivers.pop()
					
					if server_comm.rank == receiver:
					
						#I have received
						
						for a in range(q**(k-2)):
							
							#The job that the transmitter is sending
							cur_job = Blocks[transmitter//q][transmitter%q][a]
							
							#Reduce. The transmitter has already aggregated the sent batches in ONLY ONE part.
							c_reduced[cur_job-1] = c_reduced[cur_job-1] + c_rec_st3[cur_job-1][0]
							
							#test
							#Length should always be 1 since we are just receiving one aggregate
							# print(len(c_rec_st3[cur_job-1]))
	
	
	###################################################################################################################################################################################################
	#This barrier will be used for reduction time measurement by the master
	comm.Barrier()
	##################################################################################################################################################################################################

	#test
	# job = 4
	# server = 6
	# if comm.rank == server:
		# print "I am server ", comm.rank, " and for job ", job, " I have finally reduced ", c_reduced[job-1]
		
		
	#Return all reductions to master for concatenation
	master_req = [None] * (J)
	for i in range(J):
		master_req[i] = comm.Isend(c_reduced[i], dest=0, tag=20)
	MPI.Request.Waitall(master_req)