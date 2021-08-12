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
from scipy.sparse.linalg import spsolve

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


def get_finite_diff_matrix(method_name: str, lam: float, numsteps_space: int):
    diag1 = np.ones(numsteps_space - 2)
    diag2 = np.ones(numsteps_space - 1)
    if method_name == 'FE':
        # get the Forward Euler iterative timestepping matrix
        diag1 *= lam
        diag2 *= (1-2*lam)
    elif method_name == 'BE':
        diag1 *= -lam
        diag2 *= (1+2*lam)
    elif method_name == 'CN':
        A_diag1 = -0.5 * lam * diag1
        A_diag2 = (1 + lam) * diag2
        A = diags([A_diag1, A_diag2, A_diag1], [-1, 0, 1])
        B_diag1 = 0.5 * lam * diag1
        B_diag2 = (1 - lam) * diag2
        B = diags([B_diag1, B_diag2, B_diag1], [-1, 0, 1])
        return A, B
    else:
        raise NameError("Invalid input: method_name = ['FE', 'BE', 'CN']")
    # For FE and BE methods:
    mat = diags([diag1, diag2, diag1], [-1, 0, 1])
    return mat


def get_be_matrix(lam, numsteps_space):
    # get the Backward Euler implicit timestepping matrix
    be_mat = spdiags([diag1, diag2, diag1], [-1, 0, 1], numsteps_space+1, numsteps_space+1)
    return be_mat


def finite_diff(method_name: str, sol_ij, numsteps_time, fd_matrix, boundary_cond: tuple):
    sol_ij1 = np.zeros(sol_ij.shape)
    # Solve the PDE: loop over all time points
    for j in range(0, numsteps_time):
        # PDE discretised at position x[i], time t[j]
        if method_name == 'FE':
            # Forward Euler timestep at inner mesh points
            sol_ij1[1:-1] = fd_matrix.dot(sol_ij[1:-1])  # sol at next time step
        elif method_name == 'BE':
            sol_ij1[1:-1] = spsolve(fd_matrix, sol_ij[1:-1])
        else:
            raise NameError("Invalid input: method_name = ['FE', 'BE', 'CN']")

        # Boundary conditions
        sol_ij1[0] = boundary_cond[0]; sol_ij1[-1] = boundary_cond[-1]

        # Save u_j at time t[j+1]
        sol_ij[:] = sol_ij1[:]
    return sol_ij


def plot_numerical_method(space_mesh, sol, time_value, method_name: str):
    plt.plot(space_mesh, sol, 'ro', label='num')
    plt.xlabel('x')
    plt.ylabel('u(x,'+str(time_value)+')')
    plt.title(method_name)
    plt.legend(loc='best')
    plt.show()


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

# # FE METHOD:
# # get FE method matrix
# A_FE = get_finite_diff_matrix('FE', lmbda, mx)
#
# # iterate to get t=T
# u_j = finite_diff('FE', u_j, mt, A_FE, (0, 0))
#
# # Plot the final result and exact solution
# xx = np.linspace(0, L, 250)
# plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
# plot_numerical_method(x, u_j, T, 'FE Method')
# # ^FE METHOD^


# # BE METHOD:
# # get BE method matrix
# A_BE = get_finite_diff_matrix('BE', lmbda, mx)
#
# # iterate to get t=T
# u_j = finite_diff('BE', u_j, mt, A_BE, (0, 0))
#
# # Plot the final result and exact solution
# xx = np.linspace(0, L, 250)
# plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
# plot_numerical_method(x, u_j, T, 'BE Method')
# # ^BE METHOD^

# CN METHOD:
# get CN method matrices
A_CN, B_CN = get_finite_diff_matrix('CN', lmbda, mx)

sol_ij1 = np.zeros(u_j.shape)
# Solve the PDE: loop over all time points
for j in range(0, mt):
    # PDE discretised at position x[i], time t[j]
    sol_ij1[1:-1] = spsolve(A_CN, B_CN.dot(u_j[1:-1]))

    # Boundary conditions
    sol_ij1[0] = 0
    sol_ij1[-1] = 0

    # Save u_j at time t[j+1]
    u_j[:] = sol_ij1[:]

# # iterate to get t=T
# u_j = finite_diff('BE', u_j, mt, A_BE, (0, 0))

# Plot the final result and exact solution
xx = np.linspace(0, L, 250)
plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
plot_numerical_method(x, u_j, T, 'CN Method')
# ^CN METHOD^

