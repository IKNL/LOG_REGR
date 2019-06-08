import pandas as pd
import numpy as np
import os
import sys
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")


def iterate(coefficients, outcome_variable, design_matrix):
    outcome = np.matrix(design_matrix[outcome_variable]).T
    # info("Outcome shape is: " + str(outcome.shape))
    # Remove the predictor from the design matrix
    design_matrix = design_matrix.drop([outcome_variable], axis=1)
    # Calculate number of subjects available in the current study
    # (by enumerating length of ID column)
    rows_number = len(design_matrix)

    # Calculate linear predictors from observed covariate values
    # and elements of
    # current beta vector
    coefficients = np.asmatrix(coefficients)
    #info("Coefficients shape is: " + str(coefficients.shape))
    logit = np.dot(design_matrix, coefficients.T)
    #info("Logit shape is: " + str(logit.shape))
    # Apply inverse logistic transformation
    odds_ratio = np.exp(logit)
    #info("Odds ration shape is: " + str(odds_ratio.shape))
    probability = odds_ratio / (1 + odds_ratio)
    #info("Probability shape is: " + str(probability.shape))

    # Derive variance function and diagonal elements for
    # weight matrix (using squared
    # first differential of link function)
    variance = np.multiply(probability, (1 - probability))
    #info("Variance shape is: " + str(variance.shape))
    link_function_differential = 1 / (variance)
    weight_matrix = np.diagflat(variance)

    # Calculate information matrix
    info_matrix = np.dot(np.dot(design_matrix.transpose(),
                                weight_matrix), design_matrix)

    #info("Info matrix shape is: " + str(info_matrix.shape))
    # Derive u terms for score vector
    u_terms = np.multiply((outcome - probability), link_function_differential)
    # Calculate score vector
    #info("U terms shape is : " + str(u_terms.shape))
    score_vect = np.dot(np.dot(design_matrix.T, weight_matrix), u_terms)
    #info("Score vector shape is: " + str(score_vect.shape))
    # Calculate log likelihood and deviance contribution for current study
    # For convenience, ignore the element of deviance that relates to the full saturated
    # model, because that will cancel out in calculating the change in deviance from one
    # iteration to the next (Dev.total – Dev.old [see below]) because the element relating
    # to the saturated model will be the same at every iteration).
    log_likelihood = (np.dot(outcome.T, np.log(probability)) + np.dot((1 - outcome).T, np.log(1 - probability))).item(0)
    #warn(str(log_likelihood))
    deviance = -2 * log_likelihood
    # ToDO add dimensions of array if it is needed for array reconstruction
    return {
        "info_matrix": info_matrix.tolist(),
        "score_vector": score_vect.tolist(),
        "deviance": deviance,
        "rows_number": rows_number
    }


def make_local_iteration(coefficients, columns, outcome_variable, separator, local=True, data_file=""):
    if not local:
        data_file = os.environ["DATABASE_URI"]
    design_matrix = pd.read_csv(data_file, sep=separator)[columns + [outcome_variable]]
    return iterate(coefficients, outcome_variable, design_matrix)


