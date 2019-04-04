import pandas as pd
import numpy as np
import os
import sys
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

def make_local_iteration(coefficients, data_file=""):

    separator = ","
    #read database file
    if not data_file:
        node_data = pd.read_csv(os.environ["DATABASE_URI"], sep=separator)
    else:
        node_data = pd.read_csv(data_file, sep=separator)


    #Calculate number of subjects available in the current study
    #(by enumerating length of ID column)
    rows_number = len(node_data)

    #Define design matrix (matrix of covariates) to contain BMI, SNP and the
    #interaction covariate and add a column of 1s at the start for the regression constant
    design_matrix = pd.concat([pd.Series(np.repeat([1], [rows_number], axis = 0)),
                              node_data["bmi"], node_data["bmi456"],
                               node_data["snp"]], axis=1).values


    # Calculate linear predictors from observed covariate values
    # and elements of
    # current beta vector
    logit = np.dot(design_matrix, coefficients)

    # Apply inverse logistic transformation
    odds_ratio = np.exp(logit)
    probability = odds_ratio / (1 + odds_ratio)

    # Derive variance function and diagonal elements for
    # weight matrix (using squared
    # first differential of link function)
    variance = np.multiply(probability, (1 - probability))
    link_function_differential = 1 / (variance)
    weight_matrix = np.diagflat(variance)

    #Calculate information matrix
    info_matrix = np.dot(np.dot(design_matrix.transpose(),
                                weight_matrix), design_matrix)
    info("Info matrix is: ")
    info(str(info_matrix))
    #Derive u terms for score vector
    outcome = np.matrix(node_data["CC"].values)
    u_terms = np.multiply((outcome.T - probability), link_function_differential)
    #Calculate score vector
    score_vect = np.dot(np.dot(design_matrix.T, weight_matrix), u_terms)
    info("Score vector is: ")
    info(str(score_vect))
    #Calculate log likelihood and deviance contribution for current study
    #For convenience, ignore the element of deviance that relates to the full saturated
    # model, because that will cancel out in calculating the change in deviance from one
    # iteration to the next (Dev.total – Dev.old [see below]) because the element relating
    # to the saturated model will be the same at every iteration).
    log_likelihood = (np.dot(outcome, np.log(probability)) + np.dot((1 - outcome), np.log(1 - probability))).item(0)
    warn(str(log_likelihood))
    deviance = -2 * log_likelihood
    #ToDO add dimensions of array if it is needed for array reconstruction
    return{
        "info_matrix": info_matrix.tolist(),
        "score_vector": score_vect.tolist(),
        "deviance": deviance,
        "rows_number": rows_number
    }

