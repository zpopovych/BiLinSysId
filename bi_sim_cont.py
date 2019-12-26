# Continious Bi-linear system simulator and indentification tool

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Bi-linear object
# with two inputs u (dim = 2)
# and one output y  (dim = 1)
# A, B, N1, N2, C - state (structure) matrices of a system
# x - state of the system

class BiBox:
    def __init__(self, A, B, N1, N2, C, x, u):
        self.A = A
        self.B = B
        self.N1 = N1
        self.N2 = N2
        self.C = C
        self.x = x
        self.u = u
        self.y = float(C@x)
    def y(self):
        return(self.C@self.x)

# Bi-linear сcntinious object amulator
# Structure inhereted from BiBox
# Include methods to process binary signals

class BiBoxCont:
    def __init__(self,
        A = np.array([[-1, 0], [1, -2]]),
        B = np.array([[1, 0], [0, 1]]),
        N1 = np.array([[0, 0], [1, 1]]),
        N2 = np.array([[1, 1], [0, 0]]),
        C = np.array([0, 1]),
        x = np.array([[0.0], [0.0]]),
        u = np.array([[0.0], [0.0]])):
        BiBox.__init__(self, A, B, N1, N2, C, x, u)
    def Dx(self):
        dxdt = self.A@self.x + self.B@self.u + self.N1@x*self.u[0] + self.N2@self.x*self.u[1]
        return(dxdt)
    # method to process binary signal
    def  binary_signal(self, start_time = 0.0, end_time = 1.0, amp_level = 1.0, dt=0.1, u1=True, u2=True):
        A = self.A
        B = self.B
        N1 = self.N1
        N2 = self.N2
        def Dx( t, y):
            x = np.array([ [y[0]], [y[1]] ])
            dxdt = A @ x + B @ u + N1 @ x * u[0] + N2 @ x * u[1]
            res = [dxdt.flatten()[0], dxdt.flatten()[1]]
            return (res)
        if u1 and u2:
            u = np.array([[amp_level], [amp_level]])
        if u1 or u2:
            if u1: u = np.array([[amp_level], [0.0]])
            if u2: u = np.array([[0.0], [amp_level]])
        if (u1==False) and (u2==False) : u = np.array([[0.0], [0.0]])

        t_span = [start_time, end_time]
        t_eval = np.arange(start_time, end_time+dt, dt)
        x0 = [self.x.flatten()[0], self.x.flatten()[1]]
        sol = solve_ivp(lambda t, y: Dx(t,y), t_span=t_span, y0=x0, method='RK45', t_eval=t_eval)
        x = sol.y
        self.x = np.array([ [x[0, -1]], [x[1, -1]] ])
        y = np.zeros(t_eval.size)
        for i in range (0, t_eval.size):
            y[i] = self.C @ [ [x[0, i]], [x[1, i]]]
        return(t_eval, y)
    # method to process binary signal on input 1 (u1)
    def u1(self, spp=1.0, total_time = 20.0, amp_level = 1.0, dt=1.0):
        t_eval = np.arange(0.0, total_time+dt, dt)
        y = np.zeros(t_eval.size)
        spp_end = int(spp/dt)
        t_eval[0:spp_end+1], y[0:spp_end+1] = self.binary_signal(start_time = 0.0, end_time = spp, amp_level = amp_level, dt=dt, u1=True, u2=False)
        t_eval[spp_end:], y[spp_end:] = self.binary_signal(start_time = spp, end_time = total_time, amp_level = 0.0, dt=dt, u1=True, u2=False)
        return (t_eval, y)
    # method to process binary signal on input 2 (u2)
    def u2(self, spp=1.0, total_time=20.0, amp_level = 1.0, dt=1.0):
        t_eval = np.arange(0.0, total_time+dt, dt)
        y = np.zeros(t_eval.size)
        spp_end = int(spp/dt)
        t_eval[0:spp_end+1], y[0:spp_end+1] = self.binary_signal(start_time=0.0, end_time=spp, amp_level=amp_level, dt=dt,
                                                             u1=False, u2=True)
        t_eval[spp_end:], y[spp_end:] = self.binary_signal(start_time=spp, end_time=total_time, amp_level=0.0, dt=dt,
                                                           u1=False, u2=True)
        return (t_eval, y)

def inp_signal(spp, amp):
    t = np.arange(0.0, 20.0, 0.01)
    y = np.zeros(t.size)
    y[0:int(spp / 0.01)].fill(float(amp))
    return (t, y)

# Main program

# Creating new bi-linear object with standart structure

b = BiBoxCont()

#Plotting

fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(4, 2, wspace=0.5, hspace=1)

ax11 = fig.add_subplot(grid[0, 0])
ax12 = fig.add_subplot(grid[1, 0], sharey=ax11)
ax21 = fig.add_subplot(grid[2, 0], sharey=ax11)
ax22 = fig.add_subplot(grid[3, 0], sharey=ax11)

ax1 = fig.add_subplot(grid[:2, 1])
ax2 = fig.add_subplot(grid[2:, 1])


for spp in range (1,6):
    t, y = inp_signal(spp, 1)
    ax11.plot(t, y, color='green')
    ax12.plot(t, y, color='red')
    ax21.plot(t, y, color='green')
    t, y = inp_signal(spp, 0.5)
    ax22.plot(t, y, color='red')

ax11.set_title('Input signals \n (unit step)')
ax21.set_title('(different steps)')
ax22.set_xlabel('Time(sec)')

# Unit pulse
# 1st input
for i in range (1,6):
    t, y = b.u1(spp = float(i))
    ax1.plot(t, y, color='green')

#2nd input
for i in range (1,6):
    t, y = b.u2(spp = float(i))
    ax1.plot(t, y, color='red')

grn_line1 = mlines.Line2D([],[],color='green', label='1st input')
red_line1 = mlines.Line2D([],[],color='red', label='2nd input')
ax1.legend(handles=[grn_line1, red_line1])

ax1.set_title('Output signals')
ax1.set_ylabel('Step response')

# Different size inputs
# 1st input
for i in range (1,6):
    t, y = b.u1(spp = float(i))
    ax2.plot(t, y, color='green')

#2nd input
for i in range (1,6):
    t, y = b.u2(spp = float(i), amp_level = .5)
    ax2.plot(t, y, color='red')

grn_line2 = mlines.Line2D([],[],color='green', label='1st input \n(unit step)')
red_line2 = mlines.Line2D([],[],color='red', label='2nd input \n(1/2 unit step)')
ax2.legend(handles=[grn_line2, red_line2])

#ax2.set_title('(different size inputs)')
ax2.set_ylabel('Step response')
ax2.set_xlabel('Time(sec)')

#fig.savefig('BiLinCont_InpOut.png')
fig.show()


