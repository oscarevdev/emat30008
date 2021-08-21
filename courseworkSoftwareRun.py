import numpy as np
import matplotlib.pyplot as plt
from math import pi
from courseworkSoftware import shooting
from courseworkSoftware import lotka_volterra, lotka_volterra_dxdt
from courseworkSoftware import pde_initial_condition
from courseworkSoftware import run_finite_diff
from courseworkSoftware import u_I, u_exact
from courseworkSoftware import hopf_normal_form, hopf_normal_form_du1dt, hopf_normal_form_exact


def heat_equation_finite_differences(choose_method):
    """
    Run the finite differences for the heat equation

    Inputs:
    :param choose_method: Choose the finite difference method from 'FE', 'BE' or 'CN'

    Outputs:
    Plots the exact solution and solution calculated using finite differences.
    """
    # HEAT EQUATION PDE PROBLEM DEFINITION
    # simple forward Euler solver for the 1D heat equation
    #   u_t = kappa u_xx  0<x<L, 0<t<T
    # with zero-temperature boundary conditions
    #   u=0 at x=0, L, t>0
    # and prescribed initial temperature
    #   u=u_I(x) 0<=x<=L,t=0

    print("\n"+choose_method)

    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

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
    u_0 = pde_initial_condition(x, u_I, (0, 0), L)

    # Run the finite difference iterations over time span T
    u_T = run_finite_diff(choose_method, lmbda, u_0, mx, mt, (0, 0))

    # Plot the final result and exact solution
    u_exact_short = u_exact(np.linspace(0, L, len(u_T)), T, kappa, L)
    print("Finite Differences Mean Absolute Error:", np.sum(abs(u_exact_short-u_T))/len(u_T))

    xx = np.linspace(0, L, 250)
    plt.plot(xx, u_exact(xx, T, kappa, L), 'b-', label='exact')
    plt.plot(x, u_T, 'ro', label='num')
    plt.xlabel('x')
    plt.ylabel('u(x,' + str(T) + ')')
    plt.title(choose_method + ' Method')
    plt.legend(loc='best')
    plt.show()


# RUN NUMERICAL SHOOTING CODE
# Lotka-Volterra Equations
start_coords, T_lv = shooting(np.array([1,1]), 20.0, lotka_volterra, 0.2, lotka_volterra_dxdt, 0.2)
print("Lotka-Volterra Limit Cycle Coordinates:", start_coords)
print("Lotka-Volterra Period:", T_lv)

# Hopf Bifurcation
beta = 4.0
start_coords, T_hopf = shooting(np.array([1.4,1.5]), 6.0, hopf_normal_form, beta, hopf_normal_form_du1dt, beta)
print("\nHopf Limit Cycle Coordinates:", start_coords)
print("Hopf Period:", T_hopf)
hopf_exact_args = (beta, 2*pi)
print("Hopf Exact at t=0:", hopf_normal_form_exact(0, *hopf_exact_args))
print("Hopf Exact at t=2*pi:", hopf_normal_form_exact(T_hopf, *hopf_exact_args))

# Heat Equation FINITE DIFFERENCE CODE
heat_equation_finite_differences('FE')
heat_equation_finite_differences('BE')
heat_equation_finite_differences('CN')
