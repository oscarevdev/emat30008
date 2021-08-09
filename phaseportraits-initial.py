# %%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot
from math import nan

# %%
def morris_lecar(t, u, I_ext):
    # Taken from Table 1 of doi:10.1016/j.neucom.2005.03.006
    C_M = 20
    g_K = 8
    g_L = 2
    V_Ca = 120
    V_K = -80
    V_L = -60
    V_1 = -1.2
    V_2 = 18
    # Taken from Table 2 (class I) of doi:10.1016/j.neucom.2005.03.006
    g_Ca = 4.0
    phi = 1/15
    V_3 = 12
    V_4 = 17.4
    # References from doi:10.1016/j.neucom.2005.03.006
    (V, N) = u
    M_inf = 0.5*(1 + np.tanh((V - V_1)/V_2)) # (2)
    N_inf = 0.5*(1 + np.tanh((V - V_3)/V_4)) # (3)
    tau_N = 1/(phi*np.cosh((V - V_3)/(2*V_4))) # (4)
    # (1)
    dVdt = (-g_L*(V - V_L) - g_Ca*M_inf*(V - V_Ca) - g_K*N*(V - V_K) + I_ext)/C_M
    dNdt = (N_inf - N)/tau_N
    return np.array((dVdt, dNdt))

# %%

sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (0, 0))
pyplot.plot(sol.t, sol.y[0, :])

# %%

sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (0, 0))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (-20, 0))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (20, 0))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (0, 0.5))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: morris_lecar(t, u, 30), (0, 100), (0, 1.0))
pyplot.plot(sol.y[0, :], sol.y[1, :])

# %%

# V nullcline
Vval = np.linspace(-60, 20, 81)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: morris_lecar(nan, (V, N), 30)[0], 0)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan
pyplot.plot(Vval, Nval)

# N nullcline
Vval = np.linspace(-60, 20, 81)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: morris_lecar(nan, (V, N), 30)[1], 0)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan
pyplot.plot(Vval, Nval)

# %%
result = root(lambda u: morris_lecar(nan, u, 30), (-40, 0))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
result = root(lambda u: morris_lecar(nan, u, 30), (-20, 0))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
result = root(lambda u: morris_lecar(nan, u, 30), (5, 0.3))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
