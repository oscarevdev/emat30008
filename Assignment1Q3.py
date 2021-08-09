from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
import time


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


def RK4_step(tn, xn, h, fn):
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
    x_vec = np.zeros((len(t_vec),len(x1)))
    x_vec[0] = x1
    for pos in range(1,len(x_vec)):
        x_vec[pos,:] = solve_to(t_vec[pos-1],t_vec[pos],x_vec[pos-1,:], deltat, fn, method=method_name)
    return x_vec


def lin_sys(t,x):
    dxdt = x
    return dxdt


def second_order_sys(t,x):
    dxdt=np.zeros((len(x),))
    dxdt[0]=x[1]
    dxdt[1]=-x[0]
    return dxdt


# test FE one-step
x0 = np.array((1,0))
print(x0)
h = 0.5

t = np.linspace(0,10,301)

start_time=time.time()
x_euler = solve_ode(t, x0, h, second_order_sys, method_name=euler_step)
print("--- Euler Runtime: %s seconds ---" % (time.time() - start_time))

start_time=time.time()
x_RK4 = solve_ode(t, x0, h, second_order_sys, method_name=RK4_step)
print("--- RK4 Runtime: %s seconds ---" % (time.time() - start_time))

sol = odeint(second_order_sys, x0, t, tfirst=True)

# plt.plot(t, abs(sol[:,0]-x_euler), label="Euler: h="+str(h_euler))
# plt.plot(t, abs(sol[:,0]-x_RK4), label="RK4: h="+str(h_RK4))
# plt.plot(t, x_euler[:,0], label="Euler")
# plt.plot(t, x_RK4[:,0], label="RK4")
# plt.plot(t, sol[:,0], label="odeint")
plt.plot(x_euler[:,0], x_euler[:,1], label="Euler")
plt.plot(x_RK4[:,0], x_RK4[:,1], label="RK4")
plt.plot(sol[:,0], sol[:,1], label="odeint")
plt.legend()
# plt.yscale("log")
# plt.ylabel("Error")
# plt.xlabel("t")
plt.show()


