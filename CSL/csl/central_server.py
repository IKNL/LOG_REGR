from csl.node import Node
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import time
from threading import Thread


class Central_Node(Node):
    gradients_all_sites = []
    current_coefficients = []
    global_gradients = []
    second_nodes = []

    def __init__(self, outcome_variable, data=pd.DataFrame(), data_file=None):
        super().__init__(outcome_variable, data, data_file)
        self.current_coefficients = np.zeros((self.covariates.shape[1], 1))
        self.global_gradient = np.zeros((self.covariates.shape[1], 1))
        self.second_nodes = []

    # calculated using local data and using MLE function
    def get_optimized_coefficients(self):
        results = minimize(self.calculate_log_likelihood, self.current_coefficients, method='L-BFGS-B')
        return results["x"]

    def append_second_node(self, node):
        self.second_nodes.append(node)

    def calculate_log_likelihood(self, coefficients):
        logit = self.get_logit(coefficients)
        # uses formula 2 for calculations
        return (np.sum(np.log(1 + np.exp(logit))) - np.asscalar(np.dot(np.transpose(self.outcomes), logit))) / len(
            self.outcomes)

    def calculate_node_gradient(self, node, coefficients):
        self.gradients_all_sites.append(node.calculate_log_likelihood_gradient(coefficients))

    def get_node_results(self, coefficients):
        for node in self.second_nodes:
            node_calculation_thread = Thread(target=self.calculate_node_gradient, args=(node, coefficients,))
            node_calculation_thread.start()

    def calculate_global_gradient(self):
        self.gradients_all_sites = []
        # get gradients from all nodes
        self.get_node_results(self.current_coefficients)
        central_gradient = self.calculate_log_likelihood_gradient(self.current_coefficients)
        self.gradients_all_sites.append(central_gradient)
        # 1 which is added to the number of second nodes means the central server
        while len(self.gradients_all_sites) != len(self.second_nodes) + 1:
            time.sleep(0.1)
        gradients_sum = np.zeros((self.covariates.shape[1], 1))
        for node_results in self.gradients_all_sites:
            gradients_sum = np.add(gradients_sum, node_results)
        # uses part in brackets of formula (3) for calculations
        self.global_gradient = central_gradient - (gradients_sum / len(self.gradients_all_sites))

    def calculate_surrogare_likelihood(self, coefficients):
        # calculation according to formula 3
        return self.calculate_log_likelihood(coefficients) - np.asscalar(np.dot(coefficients, self.global_gradient))

    def get_vectors_difference(self, vector1, vector2):
        if len(vector1) == len(vector2):
            return pow(sum((vector1 - vector2) ** 2), 0.5)
        else:
            return np.nan

    # minimize function is used since maximized function is not present among optimization methods
    # therefore don't be surprised to see that I change the original approach
    # instead of log-likelihood maximization I minimize -log-likelihood
    def calculate_global_coefficients(self, log_file, is_odal=False, result_file=None):
        # get the best coefficients based on only central-server data
        self.current_coefficients = self.get_optimized_coefficients()
        with open(log_file, "w") as file:
            file.write("Coefficients before iterations start are: {}\n".format(self.current_coefficients))

        # it calculates the gradient term which is inside the bracket in formula (3) Take into account that it required to
        # be calculated only once
        if not is_odal:
            max_iterations = 100
            max_delta = 1e-3
            converged = False
            final_number_of_iterations = max_iterations

        with open(log_file, "a") as file:
            if is_odal:
                self.calculate_global_gradient()
                # make an update as in formula (3), gradient is saved into class variable and used inside hte formula
                # coefficients are passed as parameter because they would be optimized inside the code
                self.current_coefficients = \
                    minimize(self.calculate_surrogare_likelihood, self.current_coefficients, method='l-bfgs-b')["x"]
            else:
                for iteration in range(0, max_iterations):
                    self.calculate_global_gradient()
                    # make an update as in formula (3), gradient is saved into class variable and used inside hte formula
                    # coefficients are passed as parameter because they would be optimized inside the code
                    previous_coefficients = self.current_coefficients
                    self.current_coefficients = \
                        minimize(self.calculate_surrogare_likelihood, self.current_coefficients, method='L-BFGS-B')["x"]
                    file.write(
                        "Coefficients after iteration {} are: {}\n".format(iteration + 1, self.current_coefficients))
                    if self.get_vectors_difference(self.current_coefficients, previous_coefficients) < max_delta:
                        converged = True
                        final_number_of_iterations = iteration
                        break
        if result_file != None:
            with open(result_file, "w") as file:
                data = {}
                if not is_odal:
                    data["iterations"] = final_number_of_iterations
                    data["is_converged"] = converged
                data["coefficients"] = {}
                coefficient_index = 0
                for covariate in self.covariates.columns:
                    data["coefficients"][covariate] = \
                        self.current_coefficients[coefficient_index]
                    coefficient_index += 1
                file.write(json.dumps(data, indent=4))
