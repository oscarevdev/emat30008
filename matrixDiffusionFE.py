# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0, L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
from matplotlib import pyplot as plt
from math import pi
from scipy.sparse import diags

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


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

# Set up the solution variables
# Set initial condition
u_j = u_I(x)                  # u at current time step
u_j[0] = 0; u_j[mx] = 0
u_jp1 = np.zeros(x.size)      # u at next time step


diag1 = list(lmbda*np.ones(mx))
diag2 = list((1-2*lmbda)*np.ones(mx+1))
A_FE = diags([diag1, diag2, diag1], [-1, 0, 1]).toarray()
print(A_FE)

print(u_j)

#
# def pde_initial_condition(space_mesh, space_numsteps):
#     u_00 = np.zeros(space_mesh.size)  # u at x=0 and t=0
#     # Set initial condition
#     for i in range(0, space_numsteps + 1):
#         u_00[i] = u_I(space_mesh[i])
#     return u_00
#
#
# def finite_diff_fe(u_j):
#     # Solve the PDE: loop over all time points
#     for j in range(0, mt):
#         # Forward Euler timestep at inner mesh points
#         # PDE discretised at position x[i], time t[j]
#         u_jp1 = np.dot(A_FE, u_j)
#
#         # Boundary conditions
#         u_jp1[0] = 0; u_jp1[mx] = 0
#
#         # Save u_j at time t[j+1]
#         u_j[:] = u_jp1[:]
#     return u_j
# Solve the PDE: loop over all time points
for j in range(0, mt):
    # Forward Euler timestep at inner mesh points
    # PDE discretised at position x[i], time t[j]
    u_jp1 = np.dot(A_FE, u_j)

    # Boundary conditions
    u_jp1[0] = 0; u_jp1[mx] = 0

    # Save u_j at time t[j+1]
    u_j[:] = u_jp1[:]

print(u_j)
# Plot the final result and exact solution
plt.plot(x, u_j, 'ro', label='num')
xx = np.linspace(0, L, 250)
plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.show()
