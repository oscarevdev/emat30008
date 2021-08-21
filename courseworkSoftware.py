import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


# ODE/PDE Systems
def u_I(space_mesh: np.ndarray, space_span: float):
    """
        Get the initial space distribution for the 1D heat equation PDE:
                                         u_t = kappa u_xx  0<x<L, 0<t<T
        Input:
        :param space_mesh: space dimension of the PDE
        :param space_span: Length of the spacial domain

        Output:
        :return: Initial space distribution for the heat equation PDE
    """
    return np.sin(pi * space_mesh / space_span)


def u_exact(space_mesh: np.ndarray, time_mesh: np.ndarray, diffusion_constant: float, space_span: float):
    """
        Calculate the exact solution of the 1D heat equation PDE:
                                  u_t = kappa u_xx  0<x<L, 0<t<T
        Input:
        :param space_mesh: space dimension of the PDE
        :param time_mesh: time dimension of the PDE
        :param diffusion_constant: heat equation diffusion constant
        :param space_span: Length of the spacial domain

        Output:
        :return: Exact solution of the heat equation PDE
    """
    # the exact solution
    y = np.exp(-diffusion_constant * (pi ** 2 / space_span ** 2) * time_mesh) * np.sin(pi * space_mesh / space_span)
    return y


def lotka_volterra(t, x_vec, b):
    """Lotka-Volterra ODE equation"""
    a = 1
    d = 0.1
    x = x_vec[0]
    y = x_vec[1]
    dxdt = x*(1-x)-(a*x*y)/(d+x)
    dydt = b*y*(1-y/x)
    return np.array((dxdt, dydt))


def lotka_volterra_dxdt(t, x_vec, b):
    """First dimension of the Lotka-Volterra ODE equation"""
    return lotka_volterra(t, x_vec, b)[0]


# ODE Numerical Shooting
def shooting(u0_tilde: np.ndarray, est_T: float, dudt: callable, dudt_args, phase_condition: callable, pc_args):
    """
        Use SciPy's internal ODE solver and root finder to estimate a point on an ODE's limit cycle and its period.

        Inputs:
        :param u0_tilde: starting point for shooting (should be near limit cycle)
        :param est_T: estimated period
        :param dudt: ordinary differential equation in first order form, parameters: (t, u, arguments)
        :param dudt_args: argument values for dudt
        :param phase_condition: callable function with single float output to allow the period to be calculated,
                                                                                          parameters: (t, u, arguments)
        :param pc_args: argument values for phase_condition

        Outputs:
        :return: u_lim: point on limit cycle
                 period: limit cycle period
    """
    def periodic_bvp(g_u0_T: np.ndarray):
        """
            Define the periodic boundary value problem to be solved by the root finder.

            Inputs:
            :param g_u0_T: the estimated limit cycle start point concatenated with the estimated period
                           with the estimated period as the final value so it can be indexed with [-1]
                           this allows multidimensional inputs to be taken.
                           The start point and period need to be as one vector so the root finder
                                                                                            can solve for both values.

            Outputs:
            :return periodic_bvp_sol_eqn9: the periodic boundary value problem as detailed in equation 9
                                                                                        of the Numerical Shooting notes

        """
        # internally unpack the start point and period estimates
        g_u0 = g_u0_T[:-1]
        g_T = g_u0_T[-1]
        # create the function G(u0,T) as detailed in equation 6 of the Numerical Shooting notes
        periodic_bvp_sol = g_u0 - solve_ivp(lambda t, u: dudt(t, u, dudt_args), (0, g_T), g_u0).y[:, -1]
        # add the phase condition to get G(u0,T) as detailed in equation 9 of the Numerical Shooting notes
        periodic_bvp_sol_eqn9 = np.append(periodic_bvp_sol, phase_condition(0, g_u0, pc_args))
        return periodic_bvp_sol_eqn9

    # create the concatenated start point and period estimations
    initial_guess = np.append(u0_tilde, est_T)

    # use SciPy's internal root finding function to find the limit cycle's start point and period
    limit_cycle_vals = root(lambda u0_T: periodic_bvp(u0_T), initial_guess)

    # print the root finding results and return the solutions
    print(limit_cycle_vals)
    u_lim = limit_cycle_vals.x[:-1]
    period = limit_cycle_vals.x[-1]
    return u_lim, period


# PDE Finite Differences
def pde_initial_condition(space_mesh: np.ndarray, initial_cond_distribution: callable, boundary_cond: tuple, space_span: float):
    """
        Get the initial PDE state space

        Inputs:
        :param space_mesh: space_mesh: space dimension of the PDE
        :param initial_cond_distribution: Function giving the initial space distribution of the PDE system
        :param boundary_cond: Boundary conditions at either end of the space distribution (only 1D space dimensions allowed)
        :param space_span: Length of the spacial domain

        Outputs:
        :return u_00: numpy array of the PDE state space at t=0
    """
    # check that the space dimension is 1D
    assert len(boundary_cond) == 2
    # Set initial condition
    u_00 = initial_cond_distribution(space_mesh, space_span)
    u_00[0] = boundary_cond[0]     # boundary conditions
    u_00[-1] = boundary_cond[-1]   # boundary conditions
    return u_00


def get_finite_diff_matrix(method_name: str, lam: float, numsteps_space: int):
    """
        Get the finite difference method iterative timestepping matrix for either the Forward Euler ('FE'),
        Backwards Euler ('BE') or Crank-Nicholson ('CN') schemes.

        Inputs:
        :param method_name: Iterative timestepping scheme name, from the list: ['FE', 'BE', 'CN']
        :param lam: lambda value for the finite differences method
        :param numsteps_space: Number of values in the space dimension

        Outputs:
        :return mat: For the FE scheme return the matrix corresponding to u_{j+1} = mat*u_{j}
                     For the BE scheme return the matrix corresponding to mat*u_{j+1} = u_{j}
        :return A, B: For the Crank-Nicholson scheme return the matrices corresponding to A*u_{j+1} = B*u_{j}
    """
    # matrix/matrices will be tridiagonal so define the size
    diag1 = np.ones(numsteps_space - 2)
    diag2 = np.ones(numsteps_space - 1)
    if method_name == 'FE':
        # get the Forward Euler iterative timestepping matrix
        diag1 *= lam
        diag2 *= (1-2*lam)
        A = diags([diag1, diag2, diag1], [-1, 0, 1])
        B = None
    elif method_name == 'BE':
        # get the Backward Euler iterative timestepping matrix
        diag1 *= -lam
        diag2 *= (1+2*lam)
        A = diags([diag1, diag2, diag1], [-1, 0, 1])
        B = None
    elif method_name == 'CN':
        # get the Crank-Nicholson iterative timestepping matrices
        A_diag1 = -0.5 * lam * diag1
        A_diag2 = (1 + lam) * diag2
        A = diags([A_diag1, A_diag2, A_diag1], [-1, 0, 1])
        B_diag1 = 0.5 * lam * diag1
        B_diag2 = (1 - lam) * diag2
        B = diags([B_diag1, B_diag2, B_diag1], [-1, 0, 1])
    else:
        raise NameError("Invalid input: method_name = ['FE', 'BE', 'CN']")
    return A, B


def finite_diff(method_name: str, sol_j: np.ndarray, numsteps_time: int, boundary_cond: tuple, fd_matrix, fd_matrix2=None):
    """
        Solve the Finite Difference problem to get the solution (sol_j) of the PDE after "numsteps_time" iterations.

        Inputs:
        :param method_name: Iterative timestepping scheme name, from the list: ['FE', 'BE', 'CN']
        :param sol_j: Discretised PDE solution at time t[j]
        :param numsteps_time: Number of iterations to take in the time dimension
        :param boundary_cond: Boundary conditions at either end of the space distribution (only 1D space dimensions allowed)
        :param fd_matrix: Iterative timestepping matrix
                          FE scheme - matrix corresponding to u_{j+1} = fd_matrix*u_{j}
                          BE scheme - matrix corresponding to fd_matrix*u_{j+1} = u_{j}
                          CN scheme - matrix corresponding to fd_matrix*u_{j+1} = fd_matrix2*u_{j}
        :param fd_matrix2: Iterative timestepping matrix
                           FE scheme - None
                           BE scheme - None
                           CN scheme - matrix corresponding to fd_matrix*u_{j+1} = fd_matrix2*u_{j}

        Outputs:
        :return sol_j: Discretised PDE solution at time t[j=numsteps_time]
    """
    # define sol_j after one more timestep
    sol_j1 = np.zeros(sol_j.shape)

    # Solve the PDE: loop over all time points
    for j in range(0, numsteps_time):
        if method_name == 'FE':
            # Forward Euler timestep at inner mesh points
            sol_j1[1:-1] = fd_matrix.dot(sol_j[1:-1])  # sol at next time step
        elif method_name == 'BE':
            # Backward Euler implicit timestep at inner mesh points
            sol_j1[1:-1] = spsolve(fd_matrix, sol_j[1:-1])
        elif method_name == 'CN':
            # Crank-Nicholson implicit timestep at inner mesh points
            sol_j1[1:-1] = spsolve(fd_matrix, fd_matrix2.dot(sol_j[1:-1]))
        else:
            raise NameError("Invalid input: method_name = ['FE', 'BE', 'CN']")

        # Boundary conditions
        sol_j1[0] = boundary_cond[0]
        sol_j1[-1] = boundary_cond[-1]

        # Save sol_j at time t[j+1]
        sol_j[:] = sol_j1[:]
    return sol_j


def run_finite_diff(method_name: str, lam: float, sol_j: np.ndarray, numsteps_space: int, numsteps_time: int,
                    boundary_cond: tuple = (0, 0)):
    """
        Initiate the Finite Differences problem, create the timestepping matrix/matrices and solve the problem.
        This allows the user to simply have to change the method_name variable to get the solutions determined
        by different timestepping methods.

        Inputs:
        :param method_name: Iterative timestepping scheme name, from the list: ['FE', 'BE', 'CN']
        :param lam: lambda value for the finite differences method
        :param sol_j: Discretised PDE solution at time t[j]
        :param numsteps_space: Number of values in the space dimension
        :param numsteps_time: Number of iterations to take in the time dimension
        :param boundary_cond: Boundary conditions at either end of the space distribution (only 1D space dimensions allowed)

        Outputs:
        :return sol_T: Discretised PDE solution at time T=t[j=numsteps_time]
    """
    assert method_name in ['FE', 'BE', 'CN']
    # get the timestepping matrix
    A_fd, B_fd = get_finite_diff_matrix(method_name, lam, numsteps_space)

    # # iterate to get t=T
    sol_T = finite_diff(method_name, sol_j, numsteps_time, boundary_cond, A_fd, B_fd)

    return sol_T
