from scipy.integrate import solve_ivp
from scipy.optimize import root
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from ShootingIntro import shooting


def hopf_normal_form(t, u, beta):
    sigma = -1
    u1 = u[0]
    u2 = u[1]
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return np.array((du1dt, du2dt))


beta_model = 0.2
x0=np.array((1,1))

shooting(x0, 20, hopf_normal_form, beta_model)
