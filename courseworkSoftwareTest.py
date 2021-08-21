import unittest
import numpy as np
from courseworkSoftware import shooting
from phaseportraits_initial import morris_lecar
from courseworkSoftware import hopf_normal_form, hopf_normal_form_du1dt, hopf_normal_form_exact


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
