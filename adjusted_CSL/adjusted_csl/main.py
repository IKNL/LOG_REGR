from adjusted_csl.central_server import Central_Node
from adjusted_csl.node import Node

def calculate_logreg_csl_simulations(data, outcome_column,
                                     site_column, central_site,
                                     log_file, result_file):
    central_data = data[data[site_column] == central_site]
    del central_data["site_column"]
    central_server = Central_Node(data=central_data,
                                  id=central_site,
                                  outcome_variable=outcome_column)
    node_ids = data[site_column].unique()
    for node in node_ids:
        if node == central_site:
            continue
        node_data = data[data[site_column] == node]
        del node_data["site_column"]
        node = Node(data=node_data,
                    id=node,
                    outcome_variable=outcome_column)
        central_server.append_second_node(node)
    central_server.calculate_global_coefficients(log_file, result_file)