from adjusted_csl.node import Node
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import time
from threading import Thread
import copy


class Central_Node(Node):
    global_coefficients = []
    second_nodes = []

    def __init__(self, outcome_variable, id, data=pd.DataFrame(), data_file=None):
        super().__init__(outcome_variable, id, data, data_file)
        self.global_coefficients = np.zeros((self.covariates.shape[1], 1))
        self.global_gradient = np.zeros((self.covariates.shape[1], 1))
        self.id = id
        self.second_nodes = []

    def append_second_node(self, node):
        self.second_nodes.append(node)

    def update_node_gradients(self, coefficients, iteration):
        for node in self.second_nodes:
            node_calculation_thread = Thread(target=node.update_local_gradient, args=(coefficients, iteration, ))
            node_calculation_thread.start()

    def update_global_gradient(self, iteration):
        # get gradients from all nodes
        self.update_node_gradients(self.global_coefficients, iteration)
        self.update_local_gradient(self.global_coefficients, iteration)
        gradients_sum = np.zeros((self.covariates.shape[1], 1))
        gradients_sum = np.add(gradients_sum, self.local_gradient)
        for node in self.second_nodes:
            while node.gradient_iteration != iteration:
                time.sleep(0.0001)
            gradients_sum = np.add(gradients_sum, node.local_gradient)
        gradient_index = 0
        for covariate in self.covariates.columns:
            if "region" in covariate:
                # self.global_gradient[gradient_index] = gradients_sum[gradient_index]
                self.global_gradient[gradient_index] = 0
            else:
                self.global_gradient[gradient_index] = gradients_sum[gradient_index]\
                                                       / (len(self.second_nodes) + 1)
            gradient_index += 1

    # calculated using local data and using MLE function
    def update_central_coefficients(self):
        results = minimize(self.calculate_log_likelihood, self.global_coefficients,
                           method='L-BFGS-B', tol=1e-6)
        return results["x"] * 0

    def get_vectors_difference(self, vector1, vector2):
        if len(vector1) == len(vector2):
            return pow(sum((vector1 - vector2) ** 2), 0.5)
        else:
            return np.nan

    def update_global_coefficients(self, iteration):
        for node in self.second_nodes:
            node.global_gradient = copy.deepcopy(self.global_gradient)
            node_calculation_thread = Thread(target=node.update_local_coefficients, args=(self.global_coefficients,
                                                                                          iteration,))
            node_calculation_thread.start()
        central_coefficients = minimize(self.calculate_surrogare_likelihood, self.global_coefficients,
                                       method='L-BFGS-B', tol=1e-6)["x"]
        coefficients_sum = np.zeros((self.covariates.shape[1]))
        coefficients_sum = np.add(coefficients_sum, central_coefficients)
        for node in self.second_nodes:
            while node.coefficients_iteration != iteration:
                time.sleep(0.0001)
            coefficients_sum = np.add(coefficients_sum, node.local_coefficients)
        self.global_coefficients = coefficients_sum / (len(self.second_nodes) + 1)

    # minimize function is used since maximized function is not present among optimization methods
    # therefore don't be surprised to see that I change the original approach
    # instead of log-likelihood maximization I minimize -log-likelihood
    def calculate_global_coefficients(self, log_file, result_file=None):
        # get the best coefficients based on only central-server data
        self.global_coefficients = self.update_central_coefficients()
        with open(log_file, "w") as file:
            file.write("Coefficients before iterations start are: {}\n".format(self.global_coefficients))
        # it calculates the gradient term which is inside the bracket in formula (3) Take into account that it required to
        # be calculated only once
        max_iterations = 500
        max_delta = 1e-6
        converged = False
        final_number_of_iterations = max_iterations

        running_time = 0
        with open(log_file, "a") as file:
            start_time = time.time()
            for iteration in range(0, max_iterations):
                self.update_global_gradient(iteration)
                # make an update as in formula (3), gradient is saved into class variable and used inside hte formula
                # coefficients are passed as parameter because they would be optimized inside the code
                previous_coefficients = self.global_coefficients
                self.update_global_coefficients(iteration)
                if self.get_vectors_difference(self.global_coefficients, previous_coefficients) < max_delta:
                    running_time = time.time() - start_time
                    converged = True
                    final_number_of_iterations = iteration
                    break
                if iteration == 100 \
                        and self.get_vectors_difference(self.global_coefficients, previous_coefficients) >= 100:
                    break

        if result_file != None:
            with open(result_file, "w") as file:
                data = {}
                data["iterations"] = final_number_of_iterations
                data["is_converged"] = converged
                data["running_time"] = running_time
                data["coefficients"] = {}
                coefficient_index = 0
                for covariate in self.covariates.columns:
                    data["coefficients"][covariate] = \
                        self.global_coefficients[coefficient_index]
                    coefficient_index += 1
                file.write(json.dumps(data, indent=4))
