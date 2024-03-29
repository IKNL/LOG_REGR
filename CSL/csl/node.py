import numpy as np
import pandas as pd

class Node:
    outcome_variable = ""
    data = pd.DataFrame()
    covariates = pd.DataFrame()
    outcomes = pd.DataFrame()


    def __init__(self, outcome_variable, data = pd.DataFrame(), data_file=None):
        if data_file != None:
            self.set_data(data_file)
        else:
            self.set_data(data)
        self.outcome_variable = outcome_variable
        self.set_covariates()
        self.set_outcomes()

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
    def calculate_log_likelihood_gradient(self, coefficients):
        logit = self.get_logit(coefficients)
        logit_exp = np.exp(-logit)
        return np.dot(self.covariates.T, (1 /
                                          (1 + logit_exp)) - self.outcomes) / len(self.outcomes)