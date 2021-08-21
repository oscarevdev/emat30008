import numpy as np
import matplotlib.pyplot as plt
from courseworkSoftware import shooting
from courseworkSoftware import lotka_volterra, lotka_volterra_dxdt
from courseworkSoftware import pde_initial_condition
from courseworkSoftware import run_finite_diff
from courseworkSoftware import u_I, u_exact


# RUN NUMERICAL SHOOTING CODE
b_model = 0.2
x0 = np.array([1, 1])

start_coords, T_lv = shooting(x0, 20.0, lotka_volterra, 0.2, lotka_volterra_dxdt, 0.2)
print("Limit Cycle Start Coordinates:", start_coords)
print("Limit Cycle Period:", T_lv)

# HEAT EQUATION PDE PROBLEM DEFINITION
# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0, L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for

# RUN PDE FINITE DIFFERENCE CODE
# Set numerical parameters
mx = 10  # number of gridpoints in space
mt = 1000  # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx + 1)  # mesh points in space
t = np.linspace(0, T, mt + 1)  # mesh points in time
deltax = x[1] - x[0]  # gridspacing in x
deltat = t[1] - t[0]  # gridspacing in t
lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
print("deltax=", deltax)
print("deltat=", deltat)
print("lambda=", lmbda)

# Choose the finite difference method from 'FE', 'BE' or 'CN'
choose_method = 'CN'

# Set up the solution variables
u_0 = pde_initial_condition(x, u_I, (0, 0), L)

print(u_0)

# Run the finite difference iterations over time span T
u_T = run_finite_diff(choose_method, lmbda, u_0, mx, mt, (0, 0))

# Plot the final result and exact solution
xx = np.linspace(0, L, 250)
plt.plot(xx, u_exact(xx, T, kappa, L), 'b-', label='exact')
plt.plot(x, u_T, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,' + str(T) + ')')
plt.title(choose_method + ' Method')
plt.legend(loc='best')
plt.show()
