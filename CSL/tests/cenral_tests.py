from unittest import TestCase
from central_server import Central_Node
import numpy as np

class Central_tests(TestCase):
    def setUp(self):
        self.central = Central_Node(data_file="data/tests_data.csv", outcome_variable="outcome")


    def test_log_likelihood(self):
        coefficients = np.array([[-1], [-1]])
        log_likelihood_results = self.central.calculate_log_likelihood(coefficients)
        self.assertAlmostEqual(log_likelihood_results, -1.251, delta=1e-3)


