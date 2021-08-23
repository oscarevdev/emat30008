import unittest
import numpy as np
from math import pi
from courseworkSoftware import shooting
from courseworkSoftware import hopf_normal_form, hopf_normal_form_du1dt, hopf_normal_form_exact


class ShootingTest(unittest.TestCase):
    def test_hopf_test1(self):
        u0 = np.array([1,1])
        est_T = 6
        beta = 4
        est_limit_cycle_point, est_period = shooting(u0, est_T, hopf_normal_form, beta, hopf_normal_form_du1dt, beta)
        lim_cycle_u0 = hopf_normal_form_exact(0, beta, 2*pi)
        lim_cycle_u1 = hopf_normal_form_exact(est_period, beta, 2*pi)
        np.testing.assert_array_almost_equal(lim_cycle_u0, lim_cycle_u1, decimal=2)


if __name__ == '__main__':
    unittest.main()
