import numpy as np
from scipy import linalg

def hankel_pairs (  v = np.array([111,112,121,122,131,132]),
                    h = np.array([211,212,221,222,231,232,241,242]) ):
    v = np.asarray(v).ravel()
    h = np.asarray(h).ravel()
    a, b = np.ogrid[0:len(v), 0:len(h)]
    indx = a + b
    for i in range(1, len(v)): indx[i:, :] = indx[i:, :] + 1
    vals = np.concatenate((v, h))
    return(vals[indx[:int(len(v)/2), :]])

def hankel_Y(Y, alpha, beta):
    v = Y[ 1: alpha+1]
    h = Y[alpha+2 : alpha+beta+2]
    return hankel_pairs(v,h)

def identify_Ac(Y, alpha=5, beta=6):
    H = hankel_Y(Y,alpha=alpha, beta=beta)
    print('Size of H:', np.shape(H))
    n = np.linalg.matrix_rank(H)
    print('Rank of H:',n)
    U, s, V = linalg.svd(H, full_matrices=True)
    print('Size of U:', np.shape(U))
    print('Rank of U:', np.linalg.matrix_rank(U))
    m = 1 #number of outputs
    U_up = U[:-m, :]
    print('Size of U_up:', np.shape(U_up))
    print('Rank of U_up:', np.linalg.matrix_rank(U_up))
    U_dwn = U[m:, :]
    print('Size of U_dwn:', np.shape(U_dwn))
    print('Rank of U_dwn:', np.linalg.matrix_rank(U_dwn))

    A = U_up.transpose() @ U_dwn
    print('Size of A:', np.shape(A))
    print('Rank of A:', np.linalg.matrix_rank(A))

    Ac = np.log(A)

    return Ac

# Main program

m = 1 # m is number of outputs
r = 2 # r is number of inputs

# rank n that is the order of the state matrix A
# in our case shoud be n = 2

Y1 = np.load('ssp_1.npy')

print('Y1: \n', Y1)
print('Shape of Y1: \n', np.shape(Y1)) #(20, 2)

#choose α and β such that αm and βr are larger than or equal to n

H1 = hankel_Y(Y1,alpha=5, beta=6)

print('Hankel of Y1: H1= \n', H1)
print('Size of H1 (should be 5 x 12):', np.shape(H1))
print('Rank of H1: \n', np.linalg.matrix_rank(H1))

Ac = identify_Ac(Y1)
print('Identified Ac: \n', Ac)

