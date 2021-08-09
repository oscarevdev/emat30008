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


def shooting(u0_tilde, est_T, dudt, args):
    initial_guess = np.append(u0_tilde, est_T)
    return root(lambda u0_T: np.append(u0_T[:-1]-solve_ivp(lambda t, u: dudt(t, u, args), (0, u0_T[-1]), u0_T[:-1]).y[:, -1],
                                       dudt(0, u0_T[:-1], args)[0]), initial_guess)


# When b<0.26 there are periodic solutions
# When b>0.26 there is decay

# WHen b=0.2 and x0=[1,1]: periodic orbit w/ T=20secs (ish)

# Phase conditions (dxdt=0) when [x,y]=[0,0], [(-1 pm sqrt(41))/2, -1 pm sqrt(41))/2]

b_model = 0.2
x0=np.array((1,1))

# # plot solution
# sol = solve_ivp(lambda t, x: lotka_volterra(t,x,b_model), (0,100), x0)
# plt.plot(sol.t, sol.y[0,:])
# plt.xlabel("t")
# plt.ylabel("x")
# plt.show()

print(shooting(np.array((0.6, 0.3)), 20, lotka_volterra, 0.2))

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

for (i,x) in enumerate(x_vals):
    result = root(lambda y: lotka_volterra(math.nan, (x, y), b_model)[0], 1)
    if result.success:
        y_vals[i] = result.x
    else:
        y_vals[i] = math.nan

plt.plot(x_vals,y_vals)

# plot x nullcline
x_vals = np.linspace(-1, 1.5, 151)
y_vals = np.zeros(np.size(x_vals))

for (i,x) in enumerate(x_vals):
    result = root(lambda y: lotka_volterra(math.nan, (x, y), b_model)[1], 1)
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
