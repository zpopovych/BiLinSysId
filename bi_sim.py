# Bi-linear system simulator and indentification tool

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from math import log

# Bi-linear object emulator
# with two inputs u (dim = 2)
# and one output y  (dim = 1)
# A, B, N1, N2, C - state (structure) matrices of a system
# x - state of the system

class BiBox:
    # Class BiBox Attributes
    A = np.array([[-1, 0], [1, -2]])
    B = np.array([[ 1, 0], [0,  1]])
    N1 = np.array([[0, 0], [1,  1]])
    N2 = np.array([[1, 1], [0,  0]])
    C = np.array([0, 1])

    # Class BiBox Init
    def __init__(self, x = np.array([[1.0], [1.0]])):
        self.x = x
        self.y = float(self.C@x)

    # Class BiBox Methods
    # Single input event processing
    def tick(self, u = np.array([[0.0], [0.0]])):
        self.x = self.x + self.A@self.x+self.B@u+self.N1@self.x@u[[0]]+self.N2@self.x@u[[1]] #new state
        self.y = float(self.C@self.x) #new output
    # Sequence of input event processing
    def seq(self, inp_seq = np.array([[0.0], [0.0]])):
        if inp_seq.shape[0]!=2 :
            print("Input sequence shape mismatch!!!")
            return 0
        out_seq = np.empty(int(inp_seq.shape[1]))
        for i in range(0, int(inp_seq.shape[1])):
            self.tick(u=inp_seq[:, i].reshape((2, 1)))
            out_seq[i]=self.y
        # returning output sequence of y-s
        return out_seq

# Function to identify the bi-linear object by output sequence
def identify_A(y):
    # Hankel matrix of output sequence
    H = linalg.hankel(y)
    # Size depends on the length of input sequence
    # Size should be alpha x beta
    # where alpha = length of the input sequence ???
    # where beta = length of the output sequence ???
    print(' Hankel matrix H shape:', H.shape)
    # Rank of H should be n = order of state(structure) matrix A = 2
    # if alpha and beta are such that alpha*m and beta*r > n
    # (where m = number of outputs = 1
    #  and   r = number of inputs = 2 )
    print('H rank:', np.linalg.matrix_rank(H))
    # Using singular value decomposition of H
    U, s, V = linalg.svd(H, full_matrices=False)
    # U shoud be of dimention alpha*m x n = alpha x 2
    # V should be of dimention beta*r x n = beta*2 x 2
    print('U:', U.shape, 's:', s.shape, 'v:', V.shape)
    # deleting last m rows of U forms U_up (m=1)
    U_up  = U[ : , :-(y.size-2)]
    # deleting first m rows of U forms U_dwn (m=1)
    U_dwn = U[ : ,(y.size-2):  ]
    print('U_up:',U_up.shape, 'U_down:', U_dwn.shape)
    # Determening the state matrix
    Ac=U_up.transpose().conj()@U_dwn
    return Ac

def safe_log(x):
    if (x<1e-60):
        return log(1e-60)
    else:
        return log(x)

# Main program

# Creating new bi-linear object with standart structure
b2 = BiBox()

print ("Initial matrix A:")
print (b2.A)

# Designing input sequence

u = np.zeros((2, 5))
u[0,0:5:1].fill(2.0)
u[1,0:5:1].fill(1.0)

print ("Input sequence u:")
print(u)

# Processing response of the system

y = b2.seq(u)

# Ploting response vs time

plt.plot(y)
plt.title(str(u.shape[1])+'-sequence, '+ str(u.shape[0])+ '-inputs')
plt.show()

# Running identification function

A=identify_A(y)
vslog = np.vectorize(safe_log)

# Printing reconstructed matrix

print ("Reconstracted matrix A:")
print(vslog(A))
