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

def h_1(Y, alpha, beta):
    v = Y[ 1: alpha+1]
    h = Y[alpha+2 : alpha+beta+2]
    return hankel_pairs(v,h)

def h_k(Y, alpha, k):
    return Y[ k: alpha+k]

# Function to identify the state matrix of the system Ac

def identify_Ac(Y, alpha=5, beta=6, rnk=2, k=1):
    H = h_1(Y,alpha=alpha, beta=beta)
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

    A = np.conj(U_up.transpose()) @ U_dwn
    #print('Size of A:', np.shape(A))
    #print('Rank of A:', np.linalg.matrix_rank(A))

    Ac = logm(A)[:rnk,:rnk]



    return Ac, C, s, U

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

H1 = h_1(Y1, alpha=5, beta=6)

#print('Hankel of Y1: H1= \n', H1)
print('Size of H1 (should be 5 x 12):', np.shape(H1))
#print('Rank of H1: \n', np.linalg.matrix_rank(H1))

Ac, C, s, U1 = identify_Ac(Y1)

print('Identified \n Ac: \n', Ac)
print('C: \n', C)
print('Sigma: \n', s)

Y2 = np.load('ssp_2.npy')
H2 = h_k(Y2, alpha=5, k=2)
print('Identified \n H2: \n', H2)
print('Size of H2 (should be 5 x 2):', np.shape(H2))

B2 = np.conj(U1.transpose())@H2
print('Identified \n B2: \n', B2)
print('Size of B2:', np.shape(B2))

Y3 = np.load('ssp_3.npy')
H3 = h_k(Y3, alpha=5, k=3)
B3 = np.conj(U1.transpose())@H3
print('Identified \n B3: \n', B3)
print('Size of B3:', np.shape(B3))

Y4 = np.load('ssp_4.npy')
H4 = h_k(Y3, alpha=5, k=4)
B4 = np.conj(U1.transpose())@H4
print('Identified \n B4: \n', B4)
print('Size of B4:', np.shape(B4))

Y5 = np.load('ssp_5.npy')
H5 = h_k(Y3, alpha=5, k=5)
B5 = np.conj(U1.transpose())@H5
print('Identified \n B5: \n', B5)
print('Size of B5:', np.shape(B5))


C1 = B2[:2,:2]
C2 = B3[:2,:2] - B2[:2,:2]
C3 = B4[:2,:2] - B3[:2,:2]
C4 = B5[:2,:2] - B4[:2,:2]

rnk = 2

C1_left = C1[:,1:]
C1_right = C1[:,:-1]
A1 = C1_right @ np.conj(C1_left.transpose())
Nc1 = logm(A1[:rnk,:rnk])-Ac

C2_left = C2[:,1:]
C2_right = C2[:,:-1]
A2 = C2_right @ np.conj(C2_left.transpose())
Nc2 = logm(A2[:rnk,:rnk])-Ac


print ('Nc1 = ', Nc1)
print ('Nc2 = ', Nc2)
