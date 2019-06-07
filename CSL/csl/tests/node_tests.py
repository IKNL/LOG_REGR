from unittest import TestCase
from csl.node import Node
import numpy as np

class Node_tests(TestCase):
    def setUp(self):
        self.node = Node(data_file="data/tests_data.csv", outcome_variable="outcome")


    def test_gradient_log_likelihood(self):
        coefficients = np.array([[-1], [-1]])
        log_likelihood_results = self.node.calculate_log_likelihood_gradient(coefficients)
        self.assertEqual(len(log_likelihood_results), 2)
        self.assertAlmostEqual(log_likelihood_results[0], 0.254, delta=1e-3)
        self.assertAlmostEqual(log_likelihood_results[1], 0.254, delta=1e-3)


