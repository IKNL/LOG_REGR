import numpy as np
import pandas as pd

class Node:
    outcome_variable = ""
    data = pd.DataFrame()
    covariates = pd.DataFrame()
    outcomes = pd.DataFrame()


    def __init__(self, data_file, outcome_variable):
        self.set_data(data_file)
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

    def get_logit(self, coefficients):
        return np.dot(self.covariates, coefficients[None].T)

    # do not init coefficients with 0
    # gradient is calculate by formula below formula (3)
    # test with manual data exists
    def calculate_log_likelihood_gradient(self, coefficients):
        logit = self.get_logit(coefficients)
        logit_exp = np.exp(logit)
        return np.dot(self.covariates.T, (self.outcomes - logit_exp / (1 - logit_exp))) / len(self.outcomes)