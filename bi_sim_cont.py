# Continious Bi-linear system simulator and indentification tool

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
    def y(self):
        return(self.C@self.x)
    def Dx(self):
        dxdt = self.A@self.x + self.B@self.u + self.N1@x*self.u[0] + self.N2@self.x*self.u[1]
        return(dxdt)
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
        t_eval = np.arange(start_time, end_time, dt)
        x0 = [self.x.flatten()[0], self.x.flatten()[1]]
        sol = solve_ivp(lambda t, y: Dx(t,y), t_span=t_span, y0=x0, method='RK45', t_eval=t_eval) #something is going wrong here
        x = sol.y
        self.x = np.array([ [x[0, -1]], [x[1, -1]] ])
        y = np.zeros(t_eval.size)
        for i in range (0, t_eval.size):
            y[i] = self.C @ [ [x[0, i]], [x[1, i]]]
        return(t_eval, y)
    def u1(self, spp=1.0, total_time = 20.0, dt=1.0):
        t_eval = np.arange(0.0, total_time, dt)
        y = np.zeros(t_eval.size)
        spp_end = int(spp/dt)
        t_eval[0:spp_end], y[0:spp_end] = self.binary_signal(start_time = 0.0, end_time = spp, amp_level = 1.0, dt=dt, u1=True, u2=False)
        t_eval[spp_end-1:-1], y[spp_end-1:-1] = self.binary_signal(start_time = spp, end_time = total_time, amp_level = 0.0, dt=dt, u1=True, u2=False)
        return (t_eval, y)

    def u2(self, spp=1.0, total_time=20.0, dt=1.0):
        t_eval = np.arange(0.0, total_time, dt)
        y = np.zeros(t_eval.size)
        spp_end = int(spp / dt)
        t_eval[0:spp_end], y[0:spp_end] = self.binary_signal(start_time=0.0, end_time=spp, amp_level=1.0, dt=dt,
                                                             u1=False, u2=True)
        t_eval[spp_end-1:-1], y[spp_end-1:-1] = self.binary_signal(start_time=spp, end_time=total_time, amp_level=0.0, dt=dt,
                                                           u1=False, u2=True)
        return (t_eval, y)
b = BiBoxCont()

# 1st input
t, y1 = b.u1(spp=1.0)
plt.plot(t, y1)
t, y2 = b.u1(spp=2.0)
plt.plot(t, y2)
t, y3 = b.u1(spp=3.0)
plt.plot(t, y3)
t, y4 = b.u1(spp=4.0)
plt.plot(t, y4)
t, y5 = b.u1(spp=5.0)
plt.plot(t, y5)

#2nd input
t, y22 = b.u2(spp=2.0)
plt.plot(t, y22)
t, y23 = b.u2(spp=3.0)
plt.plot(t, y23)
t, y24 = b.u2(spp=4.0)
plt.plot(t, y24)
t, y25 = b.u2(spp=5.0)
plt.plot(t, y25)

plt.title('Binary signal')
plt.show()




