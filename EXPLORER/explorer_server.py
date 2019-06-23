import numpy as np
import time
import threading
import explorer_node
import pandas as pd

close_server = False
number_of_features = 10
clients = 10
results = []
nodes = []
nodes_data = []
is_simulation = False


def checkConverge(EP_result):
    # check if iteration finished, I believe
    if not EP_result["incoming_mw_old"]:
        return False
    for client in range(0, clients):
        distance = np.sum(abs(EP_result["incoming_mw"][client] - EP_result["incoming_mw_old"][client]))
        if distance > 1e-4:
            return False
    return True


def sendout_message(EP_result):
    if checkConverge(EP_result):
        EP_result["jobDone"] = True
        for node in nodes:
            node.stop_computations()
    central_message = {"iter": EP_result["iter"], "mw": EP_result["posterior_new_mw"],
                       "vw": EP_result["posterior_new_vw"]}
    for node in nodes:
        node.run_one_more_iteration(central_message)
    return EP_result


# TODO look into a problem with my reduced dimensionality from 2 to 1

def gaussian_multiplication(m1, v1, m2, v2):
    v = np.linalg.inv(np.linalg.inv(v1) + np.linalg.inv(v2))
    # left division in pythonr returns 3 arguments, first of them is a solution equivalent to matlab
    m = np.matmul(v, np.linalg.lstsq(v1, m1)[0] + np.linalg.lstsq(v2, m2)[0])
    return m, v


def gaussian_division(m1, v1, m2, v2):
    v = np.linalg.inv(np.linalg.inv(v1) - np.linalg.inv(v2))
    # left division in pythonr returns 3 arguments, first of them is a solution equivalent to matlab
    m = np.matmul(v, np.linalg.lstsq(v1, m1)[0] - np.linalg.lstsq(v2, m2)[0])
    return m, v


def handleIncomingMessageFromClient(EP_result):
    while (len(results) != clients):
        time.sleep(0.5)
    for client in range(0, clients):
        if (EP_result.iter == results[client]["iter"]):
            EP_result["incoming_mw_old"][client] = EP_result["incoming_mw"][client]
            EP_result["incoming_vw_old"][client] = EP_result["incoming_vw"][client]
            EP_result["incoming_mw"][client] = results[client]["mw"]
            EP_result["incoming_vw"][client] = results[client]["vw"]
    return EP_result


def combine_prior(EP_result):
    number_of_sites = clients
    # get number of features
    d = len(EP_result["posterior_new_mw"])

    prior_mean = np.ones(d) * EP_result["prior_mean"]
    prior_variance = np.eye(d) * EP_result["prior_variance"]
    prior_mean_comb = EP_result["incoming_mw"][0]
    prior_variance_comb = EP_result["incoming_vw"][0]
    for i in range(1, number_of_sites):
        prior_mean_comb, prior_variance_comb = \
            gaussian_multiplication(prior_mean_comb, prior_variance_comb,
                                    EP_result["incoming_mw"][i], EP_result["incoming_vw"][i])
        # TODO % remove n-i number of priors  what it means?
        if (number_of_sites > 1):
            prior_mean_comb, prior_variance_comb = gaussian_division \
                (prior_mean_comb, prior_variance_comb, prior_mean, prior_variance / (number_of_sites - 1));
        EP_result["posterior_new_mw"] = prior_mean_comb
        EP_result["posterior_new_vw"] = prior_variance_comb
    return EP_result


def update_all_sites(EP_result):
    if EP_result["iter"] == 0:
        # should be equal to number of features + 1 to elaborate intercept
        num_features = number_of_features
        for client in range(0, clients):
            # for each client a different matrix in the results
            EP_result["incoming_mw"].append(np.zeros(num_features))
            EP_result["incoming_vw"].append(np.eye(number_of_features))
            for feature_index in range(0, num_features):
                EP_result["incoming_mw"][client][feature_index] = np.inf
        EP_result["posterior_new_mw"] = np.ones(num_features) * EP_result["prior_mean"]
        EP_result["posterior_new_vw"] = np.eye(num_features) * EP_result["prior_variance"]
    else:
        EP_result = combine_prior(EP_result)
    EP_result["iter"] += 1
    EP_result = sendout_message(EP_result)
    # TODO why data is transposed here?
    EP_result["posterior_new_mw"] = EP_result["posterior_new_mw"].T
    return EP_result


def get_node_data(id):
    if is_simulation:
        return nodes_data[id]["covariates"], nodes_data[id]["outcome"]


def new_data_listener():
    EP_result = {}
    EP_result["clientCount"] = []
    EP_result["iter"] = 0
    EP_result["prior_variance"] = 5
    EP_result["prior_mean"] = 0
    EP_result["num_sites"] = 0
    EP_result["incoming_mw"] = []
    EP_result["incoming_vw"] = []
    EP_result["incoming_mw_old"] = []
    EP_result["incoming_vw_old"] = []
    EP_result["posterior_new_mw"] = []
    EP_result["posterior_new_vw"] = []
    EP_result["error"] = []
    EP_result["jobDone"] = False
    global close_server
    while not close_server:
        results.clear()
        for client in range(0, clients):
            node = explorer_node.explorer_node(client, nodes_data[client]["covariates"], nodes_data[client]["outcome"])
            nodes.append(node)
            thread = threading.Thread(target=node.new_job_listener,
                                      args=(results,))
            thread.start()
        if not EP_result["jobDone"]:
            EP_result = update_all_sites(EP_result)
            EP_result = handleIncomingMessageFromClient(EP_result)
            print(EP_result)
        else:
            close_server = True
    print(EP_result)


def run_simulation(data, site_column, outcome_column):
    global clients
    global is_simulation
    global nodes_data
    is_simulation = True
    #TODO modify code to preven automatic creation of intercept
    del data["intercept"]
    sites_values = data[site_column].unique()
    clients = len(sites_values)
    for site_value in sites_values:
        site_data = data[data[site_column] == site_value]
        del site_data[site_column]
        outcomes = site_data[outcome_column]
        del site_data[outcome_column]
        nodes_data.append({
            "covariates": site_data,
            "outcome": outcomes
        })
    new_data_listener()


if __name__ == '__main__':
    file_location = r"E:\simulation_0.csv"
    data = pd.read_csv(file_location)
    site_column = "site_column"
    outcome_column = "is_referred"
    run_simulation(data, site_column, outcome_column)



