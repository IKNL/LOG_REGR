from node import Node
import numpy as np
from config import info

from scipy.optimize import minimize


class Central_Node(Node):
    gradients_all_sites = []
    current_coefficients = []
    global_gradients = []

    def __init__(self, data_file, outcome_variable):
        super().__init__(data_file, outcome_variable)
        self.current_coefficients = np.zeros((self.covariates.shape[1], 1))

    # calculated using local data and using MLE function
    def get_optimized_coefficients(self):
        results = minimize(self.calculate_log_likelihood, self.current_coefficients, method='BFGS',
                           options={"disp": True})
        return results["x"]

    def calculate_log_likelihood(self, coefficients):
        logit = self.get_logit(coefficients)
        # uses formula 2 for calculations
        return (np.sum(np.log(1 + np.exp(logit))) - np.asscalar(np.dot(np.transpose(self.outcomes), logit))) / len(
            self.outcomes)

    def get_node_results(self):
        for file_number in range(0, len(info["files"])):
            node = Node(data_file=info["files"][file_number], outcome_variable=self.outcome_variable)
            self.gradients_all_sites.append(node.calculate_log_likelihood_gradient(self.current_coefficients))

    def calculate_global_gradient(self):
        # get gradients from all nodes
        self.get_node_results()
        central_gradient = self.calculate_log_likelihood_gradient(self.current_coefficients)
        self.gradients_all_sites.append(central_gradient)
        gradients_sum = np.zeros((self.covariates.shape[1], 1))
        for node_results in self.gradients_all_sites:
            gradients_sum = np.add(gradients_sum, node_results)
        # uses part in brackets of formula (3) for calculations
        return central_gradient - (gradients_sum / len(self.gradients_all_sites))

    def calculate_surrogare_likelihood(self, coefficients):
        # calculation according to formula 3
        return self.calculate_log_likelihood(coefficients) + np.asscalar(np.dot(coefficients, self.global_gradient))

    # minimize function is used since maximized function is not present among optimization methods
    # therefore don't be surprised to see that I change the original approach
    # instead of log-likelihood maximization I minimize -log-likelihood
    def get_global_coefficients(self):
        # get the best coefficients based on only central-server data
        precalculated_coefs = True
        pooled_coefs = np.array([0.0216358, 0.22843066, 0.36183711, 0.86910588, 3.50070821, 3.00057885,
                         1.82560385, 0.53803323, 0.45747878, 0.27306922, 1.1601112, -0.11292471,
                         0.62010708, 0.32833225, 0.27338948, -6.33305999])

        if precalculated_coefs:
            self.current_coefficients = pooled_coefs
        else:
            self.current_coefficients = self.get_optimized_coefficients()
        # it calculates the gradient term which is inside the bracket in formula (3) Take into account that it required to
        # be calculated only once
        self.global_gradient = self.calculate_global_gradient()
        for iteration in range(0, 10):
            print("Current coefs are: {}".format(self.current_coefficients))
            # make an update as in formula (3), gradient is saved into class variable and used inside hte formula
            # coefficients are passed as parameter because they would be optimized inside the code
            self.current_coefficients = \
            minimize(self.calculate_surrogare_likelihood, self.current_coefficients, method='BFGS',
                     options={"disp": True})["x"]
        print(self.current_coefficients)
