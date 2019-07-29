import numpy as np
import pandas as pd
from scipy.optimize import minimize
import copy


class Node:
    outcome_variable = ""
    data = pd.DataFrame()
    covariates = pd.DataFrame()
    outcomes = pd.DataFrame()
    global_gradient = []
    local_gradient = []
    node_id = -1
    gradient_iteration = -1
    coefficients_iteration = -1
    local_coefficients = []

    def __init__(self, outcome_variable, id, data=pd.DataFrame(), data_file=None):
        if data_file != None:
            self.set_data(data_file)
        else:
            self.set_data(data)
        self.outcome_variable = outcome_variable
        self.set_covariates()
        self.set_outcomes()
        self.node_id = id

    def set_outcomes(self):
        outcomes = np.array(self.data[self.outcome_variable])
        self.outcomes = outcomes.reshape(outcomes.shape + (1,))

    def set_covariates(self):
        self.covariates = self.data.drop(self.outcome_variable, axis=1)

    def set_data(self, data_file):
        self.data = pd.read_csv(data_file)

    def set_data(self, data):
        self.data = data

    def get_logit(self, coefficients):
        # None is added to be able to transpose the vector
        return np.dot(self.covariates, coefficients[None].T)

    # do not init coefficients with 0
    # gradient is calculate by formula below formula (3)
    # test with manual data exists
    def update_local_gradient(self, coefficients, iteration):
        logit = self.get_logit(coefficients)
        logit_exp = np.exp(-logit)
        self.local_gradient = np.dot(self.covariates.T, (1 /
                                                         (1 + logit_exp)) - self.outcomes) / len(self.outcomes)
        self.gradient_iteration = iteration

    def calculate_log_likelihood(self, coefficients):
        logit = self.get_logit(coefficients)
        # uses formula 2 for calculations
        return (np.sum(np.log(1 + np.exp(logit))) - np.asscalar(np.dot(np.transpose(self.outcomes), logit))) \
               / len(self.outcomes)

    def get_gradient_coefficient_product(self, coefficients):
        coefficient_index = 0
        product = 0
        for covariate in self.covariates.columns:
            if "region" not in covariate:
                product += \
                    ((self.local_gradient[coefficient_index] - self.global_gradient[coefficient_index])
                     * coefficients[coefficient_index]).item()
            coefficient_index += 1
        return product

    def calculate_surrogare_likelihood(self, coefficients):
        # calculation according to formula 3
        return self.calculate_log_likelihood(coefficients) - self.get_gradient_coefficient_product(coefficients)

    def update_local_coefficients(self, coefficients, iteration):
        coefficients = list(coefficients)
        optimization_results = minimize(self.calculate_surrogare_likelihood,
                                        coefficients, method='L-BFGS-B', tol=1e-6)
        self.local_coefficients = optimization_results["x"]
        self.coefficients_iteration = iteration
