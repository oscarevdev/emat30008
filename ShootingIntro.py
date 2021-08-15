from scipy.integrate import solve_ivp
from scipy.optimize import root
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def lotka_volterra(t, x_vec, b):
    a = 1
    d = 0.1
    x = x_vec[0]
    y = x_vec[1]
    dxdt = x*(1-x)-(a*x*y)/(d+x)
    dydt = b*y*(1-y/x)
    return np.array((dxdt, dydt))


def lotka_volterra_dxdt(t, x_vec, b):
    return lotka_volterra(t, x_vec, b)[0]


def shooting(u0_tilde: np.ndarray, est_T: float, dudt: callable, dudt_args, phase_condition: callable, pc_args):
    """
    Use SciPy's internal ODE solver and root finder to estimate a point on an ODE's limit cycle and its period.
    ------------------------------------------------
    Inputs:
    :param u0_tilde: starting point for shooting (should be near limit cycle)
    :param est_T: estimated period
    :param dudt: ordinary differential equation in first order form, parameters: (t, u, arguments)
    :param dudt_args: argument values for dudt
    :param phase_condition: callable function with single float output to allow the period to be calculated, parameters: (t, u, arguments)
    :param pc_args: argument values for phase_condition
    ------------------------------------------------
    Outputs:
    :return: u_lim: point on limit cycle
             period: limit cycle period
    """
    def periodic_bvp(g_u0_T: np.ndarray):
        """
        Define the periodic boundary value problem to be solved by the root finder.
        ------------------------------------------------
        Inputs:
        :param g_u0_T: the estimated limit cycle start point concatenated with the estimated period
                       with the estimated period as the final value so it can be indexed with [-1]
                       this allows multidimensional inputs to be taken.
                       The start point and period need to be as one vector so the root finder can solve for both values.
        ------------------------------------------------
        Outputs:
        :return periodic_bvp_sol_eqn9: the periodic boundary value problem as detailed in equation 9 of the Numerical Shooting notes
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


# When b<0.26 there are periodic solutions
# When b>0.26 there is decay

# WHen b=0.2 and x0=[1,1]: periodic orbit w/ T=20secs (ish)

# Phase conditions (dxdt=0) when [x,y]=[0,0], [(-1 pm sqrt(41))/2, -1 pm sqrt(41))/2]

b_model = 0.2
x0=np.array((1,1))

# plot solution
# sol = solve_ivp(lambda t, x: lotka_volterra(t,x,b_model), (0,100), x0)
# plt.plot(sol.t, sol.y[0,:])
# plt.xlabel("t")
# plt.ylabel("x")
# plt.show()

start_coords, T_lv = shooting(x0, 20.0, lotka_volterra, 0.2, lotka_volterra_dxdt, 0.2)
print("Limit Cycle Start Coordinates:", start_coords)
print("Limit Cycle Period:", T_lv)

# PHASE PORTRAITS
plt.plot(sol.y[0,:], sol.y[1,:])

x0=np.array((0,0))
sol = solve_ivp(lambda t, x: lotka_volterra(t,x,b_model), (0,100), x0)
plt.plot(sol.y[0,:], sol.y[1,:])

x0=np.array((0.5,0.5))
sol = solve_ivp(lambda t, x: lotka_volterra(t,x,b_model), (0,100), x0)
plt.plot(sol.y[0,:], sol.y[1,:])

x0=np.array((2,2))
sol = solve_ivp(lambda t, x: lotka_volterra(t,x,b_model), (0,100), x0)
plt.plot(sol.y[0,:], sol.y[1,:])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# ^PHASE PORTRAITS^

# NULLCLINES
# plot y nullcline
x_vals = np.linspace(-1, 1.5, 151)
y_vals = np.zeros(np.size(x_vals))

for (i,x_val) in enumerate(x_vals):
    result = root(lambda y: lotka_volterra(math.nan, (x_val, y), b_model)[0], 1)
    if result.success:
        y_vals[i] = result.x
    else:
        y_vals[i] = math.nan

plt.plot(x_vals,y_vals)

# plot x nullcline
x_vals = np.linspace(-1, 1.5, 151)
y_vals = np.zeros(np.size(x_vals))

for (i,x_val) in enumerate(x_vals):
    result = root(lambda y: lotka_volterra(math.nan, (x_val, y), b_model)[1], 1)
    if result.success:
        y_vals[i] = result.x
    else:
        y_vals[i] = math.nan
plt.plot(x_vals,y_vals)

plt.show()
# ^NULLCLINES^

# FIND EQUILIBRIA
# estimate equilbria
est_eq = root(lambda x: lotka_volterra(math.nan, x, b_model), np.array((0.3, 0.3)))
if est_eq.success:
    print("Equilibrium at {}".format(est_eq.x))
else:
    print("Failed to converge")

est_eq = root(lambda x: lotka_volterra(math.nan, x, b_model), np.array((-0.1, -0.01)))
if est_eq.success:
    print("Equilibrium at {}".format(est_eq.x))
else:
    print("Failed to converge")

est_eq = root(lambda x: lotka_volterra(math.nan, x, b_model), np.array((1.1, 0.01)))
if est_eq.success:
    print("Equilibrium at {}".format(est_eq.x))
else:
    print("Failed to converge")
# ^FIND EQUILIBRIA^

# # x0_phase_cond=(-1+(41**0.5))/2
# x0_phase_cond=1
# x0 = np.array((x0_phase_cond,x0_phase_cond))
