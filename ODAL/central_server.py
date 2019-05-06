from node import Node
import numpy as np
from config import info

from scipy.optimize import minimize


class Central_Node(Node):
    nodes_results = []
    current_coefficients = []
    global_gradients = []
    
    def __init__(self, data_file, outcome_variable):
        super().__init__(data_file, outcome_variable)
        self.current_coefficients = np.zeros((self.covariates.shape[1], 1))

    def get_optimized_coefficients(self):
        results = minimize(self.get_likelihood_negative_sign, self.current_coefficients, method='BFGS', options={"disp": True})
        return results["x"]

    def get_likelihood_negative_sign(self, coefficients):
        likelihood = self.calculate_log_likelihood(coefficients)
        return -likelihood

    def calculate_log_likelihood(self, coefficients):
        logit = self.get_logit(coefficients)
        return (np.asscalar(np.dot(np.transpose(self.outcomes), logit)) - np.sum(np.log(1 + np.exp(logit)))) \
               / len(self.outcomes)

    def get_node_results(self):
        for file_number in range(0, len(info["files"])):
            node = Node(data_file=info["files"][file_number], outcome_variable=self.outcome_variable)
            self.nodes_results.append(node.calculate_log_likelihood_gradient(self.current_coefficients))

    def calculate_global_gradient(self):
        self.get_node_results()
        nodes_likelihood_sum = np.zeros((len(self.nodes_results[0]), 1))
        for node_results in self.nodes_results:
            nodes_likelihood_sum = np.add(nodes_likelihood_sum, node_results)
        return nodes_likelihood_sum / len(self.nodes_results) \
               - self.calculate_log_likelihood_gradient(self.current_coefficients)

    def calculate_surrogare_likelihood(self, coefficients):
        return np.dot(coefficients.T, self.global_gradient) + self.calculate_log_likelihood(coefficients)

    def get_negative_surrogate_likelihood(self, coefficients):
        return -self.calculate_surrogare_likelihood(coefficients)

    def get_global_coefficients(self):
        new_coefficients = self.get_optimized_coefficients()
        self.current_coefficients = new_coefficients.reshape(new_coefficients.shape + (1,))
        self.global_gradient = self.calculate_global_gradient()
        for iteration in range(0, 10):
            updated_coefficients = minimize(self.get_negative_surrogate_likelihood, self.current_coefficients, method='BFGS',
                                            options={"disp": True})["x"]
            self.current_coefficients = updated_coefficients.reshape(updated_coefficients.shape + (1,))
            print(self.current_coefficients)
