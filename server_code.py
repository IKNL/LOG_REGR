import numpy as np
import sys
import os
import json
import time
import math

from pytaskmanager.node.FlaskIO import ClientContainerProtocol

# loggers
info = lambda msg: sys.stdout.write("info > "+msg+"\n")
warn = lambda msg: sys.stdout.write("warn > "+msg+"\n")

def calculate_coefficients(token, beta = [0, 0, 0, 0]):

    info("Setup server communication client")
    client = ClientContainerProtocol(
        token=token,
        host=os.environ["HOST"],
        port=os.environ["PORT"],
        path=os.environ["API_PATH"]
    )

    iterations_number = 0
    epsilon = math.pow(10, -8)
    max_iterations = math.pow(10, 10)
    deviance_previous_iteration = math.pow(10, 10)
    deviance = 0

    while iterations_number == 0 or \
            (abs(deviance - deviance_previous_iteration) > epsilon and
             iterations_number < max_iterations):

        deviance_previous_iteration = deviance
        iterations_number += 1

        # define the input for the summary algorithm
        info("Defining input paramaeters")
        input_ = {
            "method": "summary",
            "args": [],
            "kwargs": {
                "coefficients": beta
            }
        }

        # collaboration and image is stored in the key, so we do not need
        # to specify these
        info("Creating node tasks")
        task = client.create_new_task(input_)

        # wait for all results
        # TODO subscribe to websocket, to avoid polling
        task_id = task.get("id")
        task = client.request(f"task/{task_id}")
        while not task.get("complete"):
            task = client.request(f"task/{task_id}")
            info("Waiting for results")
            time.sleep(1)

        info("Obtaining results")
        results = client.get_results(task_id=task.get("id"))
        results = [json.loads(result.get("result")) for result in results]


        #TODO check how summation works on matrices
        info_matrix = sum([node_result["info_matrix"] for node_result in results])
        score_vector = sum([node_result["score_Vector"] for node_result in results])
        deviance = sum([node_result["deviance"] for node_result in results])
        rows = sum([node_result["rows"] for node_result in results])

        variance_covariance_matrix = np.linalg.inv(info_matrix)
        beta = variance_covariance_matrix * np.asmatrix(score_vector).T
        info("SUMMARY OF MODEL AFTER ITERATION " + iterations_number)
        info("deviance =" + deviance + "on " +
            (rows - len(beta)) + " degrees of freedom")
        info("deviance after previous iteration = "
             + deviance_previous_iteration)
        info("Variance covariance matrix is equal to ")
        info(info_matrix)
        info("Score vector is equal to")
        info(score_vector)
        info("beta is " + beta + "\n")

    info("Final beta =" + beta)
    info("Total number of iterations is" + iterations_number)
    return beta

