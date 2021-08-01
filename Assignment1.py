from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt


def euler_step(tn, xn, h, fn):
    """
    takes a single time step
    Input:
        tn -- current time
        xn -- state at time tn
        h  -- size of time step
        fn -- time derivative function in form f(x,t)
    Output:
        x -- state at time tn+h
        f -- time derivative at tn+h
    """
    x = xn+h*fn(tn, xn)
    return x


def RK4(tn, xn, h, fn):
    k1 = fn(tn, xn)
    k2 = fn(tn + h/2, xn + h*(k1/2))
    k3 = fn(tn + h/2, xn + h*(k2/2))
    k4 = fn(tn + h, xn + h*k3)
    x = xn + (1/6)*h*(k1+2*k2+2*k3+k4)
    return x


def solve_to(t1,t2,x1,deltat_max,dfdx,method=euler_step):
    # calculate the number of iterations
    num_steps = math.ceil((t2-t1)/deltat_max)
    # initiate iterable vars
    xn = x1
    tn = t1
    h = deltat_max
    for step in range(0,num_steps):
        if t2-tn < deltat_max:
            h = t2-tn
        xn = method(tn, xn, h, dfdx)
        tn = tn + h
    return xn


def solve_ode(t_vec, x1, deltat, fn, method_name=euler_step):
    x_vec = np.zeros(t_vec.shape)
    x_vec[0] = x1
    for pos in range(1,len(x_vec)):
        x_vec[pos] = solve_to(t_vec[pos-1],t_vec[pos],x_vec[pos-1], deltat, fn, method=method_name)
    return x_vec


def q1sys(t,x):
    dxdt = x
    return dxdt


# test FE one-step
x0 = 1
h = 0.5

t=np.arange(0,1,0.01)
x=solve_ode(t,x0,h, q1sys, method_name=RK4)
plt.plot(t,x)
plt.show()
