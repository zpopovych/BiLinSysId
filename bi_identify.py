# Uses data prepared by bi_sim_cont_save.py stored in the same directory
# Loads file ssp_1.npy

import numpy as np
from scipy import linalg
from scipy.linalg import logm, expm

# Function to build Hankel marix of the horiszontal pairs of numbers

def hankel_pairs ( v, h ):
    v = np.asarray(v).ravel()
    h = np.asarray(h).ravel()
    a, b = np.ogrid[0:len(v), 0:len(h)]
    indx = a + b
    for i in range(1, len(v)): indx[i:, :] = indx[i:, :] + 1
    vals = np.concatenate((v, h))
    return(vals[indx[:int(len(v)/2), :]])

# Function to build Hankel matrix of the array of pairs

def hankel_Y(Y, alpha, beta):
    v = Y[ 1: alpha+1]
    h = Y[alpha+2 : alpha+beta+2]
    return hankel_pairs(v,h)

# Function to identify the state matrix of the system Ac

def identify_Ac(Y, alpha=5, beta=6, rnk=2):
    H = hankel_Y(Y,alpha=alpha, beta=beta)
    #print('Size of H:', np.shape(H))
    n = np.linalg.matrix_rank(H)
    #print('Rank of H:',n)
    U, s, V = linalg.svd(H, full_matrices=True)
    C = U[0,:rnk]
    #print('Size of U:', np.shape(U))
    #print('Rank of U:', np.linalg.matrix_rank(U))
    m = 1 #number of outputs
    U_up = U[:-m, :rnk+1]
    #print('Size of U_up:', np.shape(U_up))
    #print('Rank of U_up:', np.linalg.matrix_rank(U_up))
    U_dwn = U[m:, :rnk+1]
    #print('Size of U_dwn:', np.shape(U_dwn))
    #print('Rank of U_dwn:', np.linalg.matrix_rank(U_dwn))

    A = U_up.transpose() @ U_dwn
    #print('Size of A:', np.shape(A))
    #print('Rank of A:', np.linalg.matrix_rank(A))

    Ac = logm(A)[:rnk,:rnk]



    return Ac, C, s

# Main program

m = 1 # m is number of outputs
r = 2 # r is number of inputs

# rank n that is the order of the state matrix A
# in our case shoud be n = 2

Y1 = np.load('ssp_1.npy')

#print('Y1: \n', Y1)
#print('Shape of Y1: \n', np.shape(Y1)) #(20, 2)

#choose α and β such that αm and βr are larger than or equal to n
alpha=5
beta=6
#print('α*m= \n', alpha*m)
#print('β*r= \n', beta*r)

H1 = hankel_Y(Y1,alpha=5, beta=6)

#print('Hankel of Y1: H1= \n', H1)
print('Size of H1 (should be 5 x 12):', np.shape(H1))
#print('Rank of H1: \n', np.linalg.matrix_rank(H1))

Ac, C, s  = identify_Ac(Y1)

print('Identified \n Ac: \n', Ac)
print('C: \n', C)
print('Sigma: \n', s)

