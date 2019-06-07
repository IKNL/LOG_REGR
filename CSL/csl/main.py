from csl.central_server import Central_Node
from csl.node import Node
import pandas as pd
import random
import os
from pathlib import Path

def calculate_logreg_csl():
    number_of_selections = 100
    initial_seed = 10
    random.seed(initial_seed)
    seeds = random.sample(range(1, 10000), number_of_selections)
    full_data = pd.read_csv("data/full_data.csv")
    number_of_nodes = 2
    main_file_min_range = 100
    selection_increase = 100
    cwd = os.getcwd()

    for selection in range(0, number_of_selections):
        central_node_records_number = main_file_min_range + selection * selection_increase
        central_data = full_data.sample(n=central_node_records_number, random_state=seeds[selection])
        central_data.to_csv(Path(cwd + "/data/central_data.csv"), index=False)
        data_left = full_data.drop(central_data.index)
        first_file = data_left.sample(n=int(len(data_left) / number_of_nodes), random_state=seeds[selection])
        first_file.to_csv(Path(cwd + "/data/first_file.csv"), index=False)
        second_file = data_left.drop(first_file.index)
        second_file.to_csv(Path(cwd + "/data/second_file.csv"), index=False)
        central = Central_Node(data_file="data/central_data.csv", outcome_variable="3_years_death")
        log_file = Path(cwd + "/simulations/predefined_coefs/simulation{}.txt".format(selection))
        open(log_file, 'w+').close()
        with open(log_file, "a+") as file:
            file.write("Central server contains {} number of records\n".format(len(central_data)))
            file.write("First node contains {} number of records\n".format(len(first_file)))
            file.write("Second node contains {} number of records\n".format(len(second_file)))
        central.calculate_global_coefficients(log_file)


def calculate_logreg_csl_simulations(data, outcome_column,
                                     site_column, central_site,
                                     log_file, result_file, is_odal):
    central_data = data[data[site_column] == central_site]
    del central_data["site_column"]
    central_server = Central_Node(data=central_data,
                           outcome_variable=outcome_column)
    node_ids = data[site_column].unique()
    for node in node_ids:
        if node == central_site:
            continue
        node_data = data[data[site_column] == node]
        del node_data["site_column"]
        node = Node(data=node_data,
                    outcome_variable=outcome_column)
        central_server.append_second_node(node)
    central_server.calculate_global_coefficients(log_file, is_odal, result_file)