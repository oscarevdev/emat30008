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


def pde_initial_condition(space_mesh, initial_cond_distribution, boundary_cond: tuple):
    assert len(boundary_cond) == 2
    # Set initial condition
    u_00 = initial_cond_distribution(space_mesh)
    u_00[0] = boundary_cond[0]; u_00[-1] = boundary_cond[-1] # boundary conditions
    return u_00


def get_fe_matrix(lam, numsteps_space):
    # get the Forward Euler iterative timestepping matrix
    diag1 = list(lmbda*np.ones(numsteps_space))
    diag2 = list((1-2*lmbda)*np.ones(numsteps_space+1))
    fe_mat = diags([diag1, diag2, diag1], [-1, 0, 1]).toarray()
    return fe_mat


def finite_diff_fe(sol_ij, numsteps_time, fe_matrix, boundary_cond: tuple):
    # Solve the PDE: loop over all time points
    for j in range(0, numsteps_time):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        sol_ij1 = np.dot(fe_matrix, sol_ij)       # sol at next time step

        # Boundary conditions
        sol_ij1[0] = boundary_cond[0]; sol_ij1[-1] = boundary_cond[-1]

        # Save u_j at time t[j+1]
        sol_ij[:] = sol_ij1[:]

    return sol_ij


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
u_j = pde_initial_condition(x, u_I, (0, 0))

# get FE method matrix
A_FE = get_fe_matrix(lmbda, mx)

# iterate to get t=T
u_j = finite_diff_fe(u_j, mt, A_FE, (0, 0))

# Plot the final result and exact solution
plt.plot(x, u_j, 'ro', label='num')
xx = np.linspace(0, L, 250)
plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
plt.xlabel('x')
plt.ylabel('u(x,'+str(T)+')')
plt.legend(loc='upper right')
plt.show()
