import numpy as np
import sys
import os
import json
import time
import math
from threading import Thread
from newton_raphson.node_code import make_local_iteration, iterate
import pandas as pd

# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

local_nodes_results = []
columns = []
outcome_variable = ""
is_simulation = False
site_column = ""
# simulated all dataset
simulated_data = pd.DataFrame()
log_file = None

def run_distributed(client, beta):
    # define the input for the node regression
    info("Defining input paramaeters")
    input_ = {
        "method": "node_regression",
        "args": [],
        "kwargs": {
            "coefficients": beta,
            "columns": columns,
            "outcome_variable": outcome_variable
        }
    }

    # collaboration and image is stored in the key, so we do not need
    # to specify these
    info("Creating node tasks")
    task = client.create_new_task(input_)

    # wait for all results
    task_id = task.get("id")
    task = client.request(f"task/{task_id}")
    while not task.get("complete"):
        task = client.request(f"task/{task_id}")
        info("Waiting for results")
        time.sleep(1)
    results = client.get_results(task_id=task.get("id"))
    results = [json.loads(result.get("result")) for result in results]
    return results


def run_local_simulation_node(beta, data):
    node_results = iterate(beta, outcome_variable, data)
    local_nodes_results.append(json.dumps(node_results))


def run_local_node(beta, file):
    node_results = make_local_iteration(coefficients=beta, outcome_variable=outcome_variable, columns=columns,
                                        data_file=file, separator=",")
    local_nodes_results.append(json.dumps(node_results))


def run_locally(beta):
    if not is_simulation:
        with open('local_config.json') as config:
            data = json.load(config)
        for node in data["nodes"]["nodes_files"]:
            node_thread = Thread(target=run_local_node, args=(beta, node,))
            node_thread.start()
        number_of_nodes = data["nodes"]["number_of_nodes"]
    if is_simulation:
        nodes = simulated_data[site_column].unique()
        for node in nodes:
            node_data = simulated_data[simulated_data[site_column] == node]
            del node_data[site_column]
            node_thread = Thread(target=run_local_simulation_node, args=(beta, node_data,))
            node_thread.start()
        number_of_nodes = len(nodes)
    while len(local_nodes_results) < number_of_nodes:
        time.sleep(0.1)
    results = local_nodes_results.copy()
    local_nodes_results.clear()
    return [json.loads(result) for result in results]


def get_results_from_nodes(beta, client):
    if client:
        return run_distributed(client, beta)
    else:
        return run_locally(beta)


def make_iteration(beta, client):
    results = get_results_from_nodes(beta, client)
    info_matrix = sum([np.asmatrix(node_result["info_matrix"]) for node_result in results])
    score_vector = sum([np.asmatrix(node_result["score_vector"]) for node_result in results])
    deviance = sum([node_result["deviance"] for node_result in results])

    variance_covariance_matrix = np.linalg.inv(info_matrix)
    beta = (variance_covariance_matrix * np.asmatrix(score_vector)).T
    start_writing_time = time.time()
    if not log_file:
        info("Variance covariance matrix is equal to\n {}\n\n".format(info_matrix))
        info("Score vector is equal to\n {}\n\n".format(score_vector))
        info("beta update is {}\n\n".format(beta))
        info("deviance = {}\n\n".format(deviance))
    else:

        with open(log_file, "w+") as file:
            file.write("Variance covariance matrix is equal to\n {}\n\n".format(info_matrix))
            file.write("Score vector is equal to\n {}\n\n".format(score_vector))
            file.write("beta update is {}\n\n".format(beta))
            file.write("deviance = {}\n\n".format(deviance))
    not_calculating_time = time.time() - start_writing_time
    return deviance, beta, not_calculating_time


def get_client(token):
    client = ""
    if token:
        from pytaskmanager.node.FlaskIO import ClientContainerProtocol
        client = ClientContainerProtocol(
            token=token,
            host=os.environ["HOST"],
            port=os.environ["PORT"],
            path=os.environ["API_PATH"]
        )
    return client


def calculate_simulated_coefficients(result_file, file_with_logs, all_data, outcome_column, site_col):
    global log_file
    log_file = file_with_logs
    open(log_file, 'w').close()
    global outcome_variable
    outcome_variable = outcome_column
    global simulated_data
    simulated_data = all_data
    global is_simulation
    is_simulation = True
    global site_column
    site_column = site_col
    global columns
    columns = list(all_data)
    columns.remove(outcome_variable)
    columns.remove(site_column)
    beta = np.asmatrix(np.zeros((1, len(columns))))
    return calculate_coefficients(beta, result_file)


def calculate_coefficients(beta, result_file=None, token=""):
    #info("Setup server communication client")
    client = get_client(token)
    iterations_number = 0
    epsilon = math.pow(10, -6)
    max_iterations = 100
    deviance_previous_iteration = math.pow(10, 10)
    deviance = -math.pow(10, 10)
    start_time = time.time()
    # some time is spend on writing into files, I extract this time from the final to achieve more reliable result
    not_calculations_time = 0

    while iterations_number == 0 or \
            (abs(deviance - deviance_previous_iteration) > epsilon and
             iterations_number < max_iterations):
        iterations_number += 1
        deviance_previous_iteration = deviance
        writing_start_time = time.time()
        if log_file is None:
            info("SUMMARY OF MODEL AFTER ITERATION {}".format(iterations_number))
            info("deviance after previous iteration = {}".format(deviance_previous_iteration))
        else:
            with open(log_file, "a") as file:
                file.write("SUMMARY OF MODEL AFTER ITERATION {}".format(iterations_number))
                file.write("deviance after previous iteration = {}"
                           .format(deviance_previous_iteration))
        not_calculations_time += time.time() - writing_start_time
        deviance, beta_update, iteration_not_calculation_time = make_iteration(beta, client)
        not_calculations_time += iteration_not_calculation_time
        beta += beta_update
    running_time = time.time() - start_time - not_calculations_time
    beta = np.array(beta.T)
    if result_file is not None:
        with open(result_file, "w+") as file:
            data = {}
            data["iterations"] = iterations_number
            data["is_converged"] = True if (iterations_number != max_iterations) else False
            data["running_time"] = running_time
            data["coefficients"] = {}
            coefficient_index = 0
            for column_index in range(len(columns)):
                data["coefficients"][columns[column_index]] = beta[column_index][0]
                coefficient_index += 1
            file.write(json.dumps(data, indent=4))
    return beta
