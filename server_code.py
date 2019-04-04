import numpy as np
import sys
import os
import json
import time
import math
from threading import Thread
from node_code import make_local_iteration


# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

local_nodes_results = []


def run_distributed(client, beta):
    # define the input for the node regression
    info("Defining input paramaeters")
    input_ = {
        "method": "node_regression",
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
    results = client.get_results(task_id=task.get("id"))
    results = [json.loads(result.get("result")) for result in results]
    return results


def run_local_node(beta, file):
    node_results = make_local_iteration(beta, file)
    local_nodes_results.append(json.dumps(node_results))


def run_locally(beta):
    with open('local_config.json') as config:
        data = json.load(config)
    for node in data["nodes"]["nodes_files"]:
        node_thread = Thread(target=run_local_node, args=(beta, node,))
        node_thread.start()
    number_of_nodes = data["nodes"]["number_of_nodes"]
    while len(local_nodes_results) < number_of_nodes:
        time.sleep(1)
    results = local_nodes_results.copy()
    local_nodes_results.clear()
    return [json.loads(result) for result in results]


def get_results_from_nodes(beta, client):
    if client:
        return run_distributed(client, beta)
    else:
        return run_locally(beta)


def make_iteration(beta, client):
    info("Obtaining results")
    results = get_results_from_nodes(beta, client)
    # TODO check how summation works on matrices
    info_matrix = sum([np.asmatrix(node_result["info_matrix"]) for node_result in results])
    score_vector = sum([np.asmatrix(node_result["score_vector"]) for node_result in results])
    deviance = sum([node_result["deviance"] for node_result in results])
    rows = sum([node_result["rows_number"] for node_result in results])

    variance_covariance_matrix = np.linalg.inv(info_matrix)
    beta = variance_covariance_matrix * np.asmatrix(score_vector)
    info("Variance covariance matrix is equal to ")
    info(str(info_matrix))
    info("Score vector is equal to")
    info(str(score_vector))
    info("beta update is " + str(beta) + "\n")
    info("deviance =" + str(deviance) + "on " +
         str((rows - len(beta))) + " degrees of freedom")

    return deviance, beta


def get_client(token):
    client=""
    if token:
        from pytaskmanager.node.FlaskIO import ClientContainerProtocol
        client = ClientContainerProtocol(
            token=token,
            host=os.environ["HOST"],
            port=os.environ["PORT"],
            path=os.environ["API_PATH"]
        )
    return client


def calculate_coefficients(token="", beta=np.asmatrix([.0, .0, .0, .0]).T):
    info("Setup server communication client")
    client = get_client(token)
    iterations_number = 0
    epsilon = math.pow(10, -8)
    max_iterations = 10
    deviance_previous_iteration = math.pow(10, 10)
    deviance = -math.pow(10, 10)

    while iterations_number == 0 or \
            (abs(deviance - deviance_previous_iteration) > epsilon and
             iterations_number < max_iterations):
        iterations_number += 1
        deviance_previous_iteration = deviance
        info("SUMMARY OF MODEL AFTER ITERATION " + str(iterations_number))
        deviance, beta_update = make_iteration(beta, client)
        beta += beta_update
        info("deviance after previous iteration = "
             + str(deviance_previous_iteration))

    info("Final beta =" + str(beta))
    info("Total number of iterations is" + str(iterations_number))
    return beta
