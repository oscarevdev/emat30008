import unittest
import numpy as np
from math import sqrt
from math import sin
from math import cos
from courseworkSoftware import shooting
from phaseportraits_initial import morris_lecar


def hopf_normal_form(t, u, beta):
    sigma = -1
    u1 = u[0]
    u2 = u[1]
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return np.array((du1dt, du2dt))


def hopf_normal_form_du1dt(t, u, beta):
    return hopf_normal_form(t, u, beta)[0]


def hopf_normal_form_exact(t, beta):
    # complete a single limit cycle oscillation of the hopf normal form
    u1 = sqrt(beta)*cos(t)
    u2 = sqrt(beta)*sin(t)
    return np.array([u1, u2])


class ShootingTest(unittest.TestCase):
    def test_hopf_test1(self):
        u0 = np.array([1,1])
        est_T = 10
        beta = 0.5
        est_limit_cycle_point, est_period = shooting(u0, est_T, hopf_normal_form, beta,
                                                     hopf_normal_form_du1dt, beta)
        beta = 0.5
        lim_cycle_u0 = hopf_normal_form_exact(0, beta)
        lim_cycle_u1 = hopf_normal_form_exact(est_period, beta)
        self.assertEqual(lim_cycle_u0, lim_cycle_u1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ShootingTest('test_hopf_test1'))
    return suite


if __name__ == '__main__':
    unittest.main()
# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     runner.run(suite())
