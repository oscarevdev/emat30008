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


def u_I(space_mesh: np.ndarray):
    """
    Input:
    :param space_mesh:

    Output:
    :return:
    """
    # initial temperature distribution
    y = np.sin(pi * space_mesh / L)
    return y


def u_exact(space_mesh: np.ndarray, time_mesh: np.ndarray):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * time_mesh) * np.sin(pi * space_mesh / L)
    return y


def pde_initial_condition(space_mesh: np.ndarray, initial_cond_distribution: callable, boundary_cond: tuple):
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


def finite_diff(method_name: str, sol_ij: np.ndarray, numsteps_time: int, fd_matrix, fd_matrix2=None, boundary_cond: tuple = (0, 0)):
    sol_ij1 = np.zeros(sol_ij.shape)
    # Solve the PDE: loop over all time points
    for j in range(0, numsteps_time):
        # PDE discretised at position x[i], time t[j]
        if method_name == 'FE':
            # Forward Euler timestep at inner mesh points
            sol_ij1[1:-1] = fd_matrix.dot(sol_ij[1:-1])  # sol at next time step
        elif method_name == 'BE':
            # Backward Euler implicit timestep at inner mesh points
            sol_ij1[1:-1] = spsolve(fd_matrix, sol_ij[1:-1])
        elif method_name == 'CN':
            # Crank-Nicholson implicit timestep at inner mesh points
            sol_ij1[1:-1] = spsolve(fd_matrix, fd_matrix2.dot(sol_ij[1:-1]))
        else:
            raise NameError("Invalid input: method_name = ['FE', 'BE', 'CN']")

        # Boundary conditions
        sol_ij1[0] = boundary_cond[0]; sol_ij1[-1] = boundary_cond[-1]

        # Save u_j at time t[j+1]
        sol_ij[:] = sol_ij1[:]
    return sol_ij


def run_finite_diff(method_name: str, lam: float, sol_ij: np.ndarray, numsteps_space: int, numsteps_time: int,
                    boundary_cond: tuple = (0, 0)):

    assert method_name in ['FE', 'BE', 'CN']

    # get the stepping matrix
    if method_name != 'CN':
        A_fd = get_finite_diff_matrix(method_name, lam, numsteps_space)
        B_fd = None
    else:
        A_fd, B_fd = get_finite_diff_matrix(method_name, lam, numsteps_space)

    # # iterate to get t=T
    sol_T = finite_diff(method_name, sol_ij, numsteps_time, A_fd, B_fd, boundary_cond)

    return sol_T


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
u_0 = pde_initial_condition(x, u_I, (0, 0))

# Run the finite difference iterations over time span T
u_T = run_finite_diff(choose_method, lmbda, u_0, mx, mt, (0, 0))

# Plot the final result and exact solution
xx = np.linspace(0, L, 250)
plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
plt.plot(x, u_T, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,' + str(T) + ')')
plt.title(choose_method + ' Method')
plt.legend(loc='best')
plt.show()
